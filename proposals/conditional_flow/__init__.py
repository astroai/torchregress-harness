"""PROPOSAL #2 — Bundled :class:`ConditionalFlow` estimator (working draft).

Status
------
This package contains a working-draft implementation of the upstream
``ConditionalFlow`` class targeted at ``torchregress``.  When the upstream PR lands,
this module becomes a thin compatibility shim that re-exports the canonical
class from ``torchregress``.

Public API
----------
- :class:`ConditionalFlow` — bundles a backbone MLP, a zuko normalizing flow,
  and :class:`torchregress.losses.nflows.NormalizingFlowLoss` behind a single
  sklearn-style estimator exposing ``fit`` / ``predict`` / ``predict_interval`` /
  ``sample`` / ``log_prob`` / ``posterior`` / ``save`` / ``load``.

Cross-cutting proposals addressed by this draft
-----------------------------------------------
* :doc:`PROPOSALS.md` §2 — Bundled conditional-flow estimator
* :doc:`PROPOSALS.md` §4 — ``predict_interval`` for NF (delegated to MC quantiles)
* :doc:`PROPOSALS.md` §5 — ``posterior`` extraction (1-D histogram, n_features==1 only)
* :doc:`PROPOSALS.md` §12 — Standardized save/load (state-dict + JSON sidecar)
* :doc:`PROPOSALS.md` §13 — Auto-inferred ``context_dim`` when caller leaves ``None``

See :mod:`proposals.conditional_flow.conditional_flow` for the implementation
and ``proposals/conditional_flow/README.md`` for design rationale + port
instructions + open-questions list.
"""

from .conditional_flow import ConditionalFlow

__all__ = ["ConditionalFlow"]
