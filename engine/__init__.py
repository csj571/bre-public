"""BRE engine: reusable Bayesian primitives.

This public tree ships the **numpy/scipy-only** core. Every module here imports
torch-free, so the whole engine runs on a bare `pip install numpy scipy`.

The torch-dependent preference learner (`SomaticBayesianEngine`) that lives in
the research repo is deliberately NOT part of this package — see PROVENANCE.md.

`GaussianProcessRegressor` is still exposed lazily (PEP 562) so that importing
the package costs nothing until you touch a submodule; the changepoint, Kalman,
calibration, Beta-Binomial, policy-gate and registry modules are imported
directly:

    from engine.changepoint import BOCPD, detect_changepoints
    from engine.calibration import brier_score, ece
    from engine.policy_gate import decide
"""

__all__ = ["GaussianProcessRegressor", "set_seed"]


def __getattr__(name):
    if name == "GaussianProcessRegressor":
        from .gp import GaussianProcessRegressor
        return GaussianProcessRegressor
    if name == "set_seed":
        from .seeding import set_seed
        return set_seed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
