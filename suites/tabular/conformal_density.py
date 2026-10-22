"""Conformalized conditional-density benchmark on synthetic data.

Trains two base density estimators (NSF and a contrastive-flow positive density)
on scalar and two-target synthetic conditional densities, then evaluates raw
sample intervals, marginal CTI conformal, Mondrian target-tercile CTI, and WALDO
density regions on a clean test split.  This is the CANFAR-runnable version of
``tools/conformal_density_bench.py``.

The point is calibration methodology, not a new state-of-the-art density.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tools.conformal_density_bench import (
    _cti_metrics,
    _data,
    _Encoder,
    _metrics,
    _sample_fn,
    _train,
    _waldo_metrics,
)

from harness.devices import current_device
from harness.runners import HarnessConfig, HarnessRunner

_METHODS = (
    "NSF",
    "NSF_CTI",
    "NSF_CTI_Mondrian",
    "NSF_WALDO",
    "ContrastiveFlow",
    "ContrastiveFlow_CTI",
    "ContrastiveFlow_CTI_Mondrian",
    "ContrastiveFlow_WALDO",
)
_EPOCHS = 80
_N_FIT, _N_CAL, _N_TEST = 3000, 1200, 2000


class ConformalDensitySuite(HarnessRunner):
    @property
    def name(self) -> str:
        return "conformal_density"

    @property
    def description(self) -> str:
        return (
            "Synthetic scalar/multi-target conditional densities: raw samples vs CTI, "
            "Mondrian CTI and WALDO regions for NSF and ContrastiveFlow"
        )

    @property
    def supported_datasets(self) -> list[str]:
        return ["cd_scalar", "cd_multi"]

    def methods(self) -> list[str]:
        return list(_METHODS)

    def library_name(self, method: str) -> str:
        return "torchregress"

    def library_version(self, method: str) -> str:
        import torchregress

        return str(torchregress.__version__)

    def method_notes(self, method: str) -> str:
        return {
            "NSF": "raw sample 5/95 intervals from an NSF density",
            "NSF_CTI": "marginal split-conformal level sets via CTI",
            "NSF_CTI_Mondrian": "CTI with target-tercile Mondrian groups",
            "NSF_WALDO": "WALDO density regions from NSF samples",
            "ContrastiveFlow": "raw sample intervals from a ContrastiveFlow positive density",
            "ContrastiveFlow_CTI": "marginal CTI conformal around ContrastiveFlow density",
            "ContrastiveFlow_CTI_Mondrian": "Mondrian CTI around ContrastiveFlow density",
            "ContrastiveFlow_WALDO": "WALDO regions from ContrastiveFlow samples",
        }[method]

    def fit(self, method: str, splits: dict[str, Any], config: HarnessConfig) -> dict[str, Any]:
        del method, splits, config
        raise NotImplementedError("conformal_density uses custom run()")

    def predict(self, method: str, model_state: Any, X_test: np.ndarray, config: HarnessConfig):
        raise NotImplementedError("conformal_density uses custom run()")

    def evaluate(
        self, method: str, model_state: Any, splits: dict[str, Any], config: HarnessConfig
    ) -> dict[str, Any]:
        raise NotImplementedError("conformal_density uses custom run()")

    def run(
        self,
        dataset_name: str = "",
        config: HarnessConfig | None = None,
        splits: dict[str, np.ndarray] | None = None,
        *,
        scenario_name: str | None = None,
    ) -> list[dict[str, Any]]:
        del splits, scenario_name
        cfg = config or HarnessConfig()
        multi = dataset_name == "cd_multi"
        seed = int(cfg.seed)
        x_fit, y_fit = _data(_N_FIT, multi, seed)
        x_cal, y_cal = _data(_N_CAL, multi, seed + 1000)
        x_test, y_test = _data(_N_TEST, multi, seed + 2000)
        y_dim = 2 if multi else 1
        epochs = 3 if cfg.smoke else min(int(cfg.epochs or _EPOCHS), _EPOCHS)
        rows: list[dict[str, Any]] = []
        for base_name, contrastive in (("NSF", False), ("ContrastiveFlow", True)):
            torch.manual_seed(seed)
            enc = _Encoder(2, 64).to(current_device())
            from torchregress.losses import (
                ContrastiveFlowLoss,
                NormalizingFlowLoss,
                create_flow_model,
            )

            flow = create_flow_model(
                n_features=y_dim,
                context_dim=64,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[64, 64],
            ).to(current_device())
            loss_fn = (
                ContrastiveFlowLoss(flow=flow, temperature=1.0, margin=0.0)
                if contrastive
                else NormalizingFlowLoss(flow=flow)
            )
            _train(
                enc,
                loss_fn,
                x_fit,
                y_fit,
                epochs=epochs,
                contrastive=contrastive,
            )
            enc.eval()
            loss_fn.eval()
            with torch.no_grad():
                ctx_cal = enc(torch.as_tensor(x_cal, dtype=torch.float32, device=current_device()))
                y_cal_t = torch.as_tensor(y_cal, dtype=torch.float32, device=current_device())
                if y_cal_t.ndim == 1:
                    y_cal_t = y_cal_t.unsqueeze(-1)
            raw = _metrics(enc, loss_fn, x_test, y_test)
            raw.pop("_ctx", None)
            key_cal = y_cal.reshape(len(y_cal), -1).mean(1)
            key_test = y_test.reshape(len(y_test), -1).mean(1)
            quantiles = np.quantile(key_cal, [0.33, 0.67])
            groups_cal = np.digitize(key_cal, quantiles)
            groups_test = np.digitize(key_test, quantiles)
            cti, _ = _cti_metrics(loss_fn, ctx_cal, y_cal_t, x_test, y_test, enc)
            cti_g, _ = _cti_metrics(
                loss_fn,
                ctx_cal,
                y_cal_t,
                x_test,
                y_test,
                enc,
                groups_cal=groups_cal,
                groups_test=groups_test,
            )
            sample_fn = _sample_fn(loss_fn, enc)
            waldo = _waldo_metrics(loss_fn, sample_fn, ctx_cal, y_cal_t, x_test, y_test, enc)
            common = {
                "Library": "torchregress",
                "LibraryVersion": self.library_version(base_name),
                "Dataset": dataset_name,
                "Suite": self.name,
                "NTargets": y_dim,
                "Device": str(current_device()),
                "NLL": raw["NLL"],
                "CRPS": raw["CRPS"],
                "RMSE": None,
            }
            specs = [
                (base_name, raw["SampleCov90"], raw["SampleWidth90"], "raw samples"),
                (f"{base_name}_CTI", cti["CTICov"], cti["CTIWidth"], "marginal CTI"),
                (
                    f"{base_name}_CTI_Mondrian",
                    cti_g["CTICov"],
                    cti_g["CTIWidth"],
                    "Mondrian CTI (target terciles)",
                ),
                (f"{base_name}_WALDO", waldo["WaldoCov"], waldo["WaldoWidth"], "WALDO density"),
            ]
            for name, coverage, width, note in specs:
                rows.append(
                    {
                        **common,
                        "Method": f"{self.name}/{name}",
                        "Coverage90": float(coverage),
                        "Notes": f"{note}; width={width:.3f}"
                        if np.isfinite(width)
                        else f"{note}; width=NA",
                    }
                )
        return rows
