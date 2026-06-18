# PROPOSAL #2 — Bundled `ConditionalFlow` Estimator (Working Draft)

**Status:** Working draft awaiting port into `torchregress` upstream.
**Repo:** `proposals/conditional_flow/` in `torchregress-harness`.
**Spec:** see [`PROPOSALS.md` §2](../../PROPOSALS.md) in this repo for the
authoritative requirement text.

## Surface

```python
from proposals.conditional_flow import ConditionalFlow

flow = ConditionalFlow(n_features=1, epochs=60, seed=42)
flow.fit(X_train, y_train)                            # returns self
mean       = flow.predict(X_test)                      # (N, n_features)
lo, hi     = flow.predict_interval(X_test,            # ((N, n_features), (N, n_features))
                                   confidence=0.9)
samples    = flow.sample(X_test, n_samples=128)        # (N, 128, n_features)
log_probs  = flow.log_prob(X_test, y_test)             # (N,)
pdf        = flow.posterior(X_test, grid=grid)         # (N, len(grid))  for n_features==1
flow.save("/tmp/cf.pt")
flow_re    = ConditionalFlow.load("/tmp/cf.pt")
```

## Cross-cutting proposals addressed (in this single draft)

| Proposal | Resolution |
|---|---|
| PROPOSAL §2 — Bundled `ConditionalFlow` estimator         | ✅ this draft |
| PROPOSAL §4 — `predict_interval()` on NormalizingFlowLoss (library-side) | ⚠ wrapped on `ConditionalFlow`; `NFL.predict_interval()` library-side PR remains open |
| PROPOSAL §5 — `posterior()` extraction                   | ✅ this draft (1-D histogram-based, only `n_features==1`) |
| PROPOSAL §12 — Standardized save / load                  | ✅ this draft (state-dict + JSON sidecar) |
| PROPOSAL §13 — Auto-inferred `context_dim`              | ✅ this draft (deferred to `.fit()`, `_default_context_dim`) |

PROPOSAL §15 (unify `DistributionLoss` interface) is partially addressed:
`ConditionalFlow.log_prob(X, y)` exposes a uniform log-probability surface;
`NFL` itself still has the asymmetric method set documented in
[`PROPOSALS.md` §15](../../PROPOSALS.md).

## Decisions encoded in this draft

* **Naming:** `ConditionalFlow` (matches PROPOSAL §2 verbatim). Collides with
  normflows' `ConditionalNormalizingFlow`; we keep `ConditionalFlow` as the
  canonical torchregress name.
* **`flow_type="nsf"`** wired; MAF / RealNVP raise `ValueError` in this draft
  and are flagged as future work.
* **Bench-tested hyperparameters:** `hidden_features=(64, 64)`,
  `n_transforms=3`, default `lr=1e-3` — matches the empirical knee in
  [`PARITY_REPORT.md` §13/§14](../../PARITY_REPORT.md).
* **Y-standardization:** left to the caller (sklearn convention).
* **Coercion:** `np.asarray` only. `pd.DataFrame` support is handled at the
  harness layer per PROPOSAL §8; library classes don't need to depend on
  pandas.
* **Determinism:** per-instance `torch.Generator` captured in `fit()` so
  repeated MC draws (across `sample`/`predict`/`predict_interval`/`posterior`)
  are bit-identical regardless of intervening global-RNG consumption. Mirrors
  the harness-side `wrappers.NormalizingFlowRegressor` contract.
* **Persistence:** state-dict (`*.pt`) + JSON sidecar (`*.meta.json`). Both
  must be moved together; the load helper refuses if metadata is missing.

## Validation

```bash
# Smoke test from this repo's venv (which already has torchregress installed):
uv run python proposals/conditional_flow/conditional_flow.py
# expected: [conditional_flow._demo] OK: is_fitted=True, predict=(50, 1),
#          interval=(50, 1)x(50, 1), samples=(50, 4, 1), log_prob=(50,),
#          posterior=(3, 25)

# Or import-check only:
uv run python -c "from proposals.conditional_flow import ConditionalFlow; print('OK')"
```

Production validation is deferred to the upstream PR's `torchregress/tests/`
suite; once ported, `tests/test_wrappers.py::test_*` in the harness should
include `ConditionalFlowRegressor` parity checks against the existing
`NormalizingFlowRegressor`.

## Risks / open questions

1. **Precision** — float32 throughout. Future users may demand float64; flag
   not exposed yet.
2. **MC sample count** — `predict()` uses 256, `predict_interval` /
   `posterior` use 512. The §14 K-sweep-style tuning is downstream of this
   draft.
3. **`posterior()` smoothing** — histogram is jagged by construction. A
   `kde=True` flag (with `scipy.stats.gaussian_kde`) would smooth but adds a
   soft-dependency surface.
4. **`posterior()` for `n_features > 1`** — raises `NotImplementedError`.
   Multi-output posteriors need dimension-specific slices (e.g. marginal PDFs
   per output coordinate); follow-up PR.
5. **Predict_interval library-side**: PROPOSAL §4 still calls for
   `NormalizingFlowLoss.predict_interval()` itself; this draft only wires
   `predict_interval` on `ConditionalFlow`. Differences are small for
   end-users but large for direct `DistributionLoss` consumers.
6. **`create_flow_model` `flow_type` coverage** — only `"nsf"` is wired;
   `zuko` supports `MAF`, `RealNVP`, `GF` etc. — extend via a dispatch table
   once zuko version constraints are locked.
7. **`flow_type` validation** — currently raises `ValueError` for
   non-`"nsf"` flows at `__init__` time. Could relax and route via
   `create_flow_model`'s own error path so forward compatibility is preserved.

## Port instructions

When this draft is approved, porting into `torchregress` upstream means:

1. **Move** `proposals/conditional_flow/conditional_flow.py` into
   `torchregress/losses/` (or a new `torchregress/flows/` submodule).
2. **Re-export** `ConditionalFlow` from `torchregress/__init__.py` (top-level
   class for one-line construction per the public API).
3. **Update** `DistributionLoss` protocol in `losses/base.py` to include
   `predict_interval(self, y_pred, confidence=0.9, n_samples=512)` (PROPOSAL §4).
4. **Add tests** in `tests/test_flows.py` covering: scalar/vector targets,
   `posterior` shape + boundary handling, save/load round-trip,
   determinism guards (bit-identical MC draws across calls).
5. **Optionally deprecate** the inline 3-object wiring in the harness's
   `wrappers.NormalizingFlowRegressor._build_model` once `ConditionalFlow` is
   the canonical path.

No new third-party dependencies are required — the draft relies only on
already-imported `NormalizingFlowLoss`, `create_flow_model`, zuko, and torch.

## Comparison vs pzflow & normflows

| Library | One-line class | `predict_interval` | `posterior` | `save`/`load` | `fit/predict` sklearn-style |
|---|---|---|---|---|---|
| **torchregress (new)** | `ConditionalFlow(n_features=...)` | ✅ (this draft) | ✅ (this draft, scalar only) | ✅ (this draft) | ✅ (this draft) |
| pzflow                    | `pzflow.Flow(...)`                | via `Flow.posterior` | via `Flow.posterior(inputs, column, grid)` | ✅ `flow.save(file)` / `Flow.load(file)` | ⚠ `.train(df, epochs=...)` (DataFrame-centric, not sklearn) |
| normflows                 | `ConditionalNormalizingFlow(...)` | ❌                            | ❌                                          | ⚠ requires manual `state_dict` plumbing | ✅ |

`ConditionalFlow` picks up the best of each: pzflow's one-line API,
normflows' sklearn-ish interface, and torchregress-native `NormalizingFlowLoss`
log_prob / sample / quantile plumbing.
