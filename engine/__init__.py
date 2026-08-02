"""BRE engine: reusable Bayesian primitives.

Public names are exposed lazily (PEP 562) so that importing a numpy-only
primitive does NOT pull in torch via the Bayesian preference learner. Every
module except `preference` imports torch-free, so the calibration, changepoint
and GP code runs on a bare `pip install numpy scipy`.

    from engine.changepoint import BOCPD, detect_changepoints
    from engine.calibration import brier_score, ece
    from engine.policy_gate import decide

`preference` needs torch — install the extra: `pip install "bre-engine[torch]"`.
"""

__all__ = ["GaussianProcessRegressor", "BayesianPreferenceLearner", "set_seed"]


def __getattr__(name):
    if name == "GaussianProcessRegressor":
        from .gp import GaussianProcessRegressor
        return GaussianProcessRegressor
    if name == "BayesianPreferenceLearner":
        from .preference import BayesianPreferenceLearner
        return BayesianPreferenceLearner
    if name == "set_seed":
        from .seeding import set_seed
        return set_seed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
