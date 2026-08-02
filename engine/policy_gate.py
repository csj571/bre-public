"""Three-action policy gate.

The decision layer: given a posterior and its epistemic/aleatoric uncertainty
split, do exactly one of ACT / QUERY MORE / DEFER. This is the safety valve —
abstain (DEFER) rather than bluff. Pure decision-theoretic threshold rule.
"""
from enum import Enum


class Action(str, Enum):
    ACT = "ACT"               # confident enough — surface the recommendation
    QUERY_MORE = "QUERY_MORE"  # uncertain but reducibly so — gather one more datum
    DEFER = "DEFER"           # irreducible / sensitive — hand to the human


def decide(epistemic_std: float, aleatoric_std: float, confidence: float, *,
           eig: float = None, sensitive: bool = False,
           eps_epistemic: float = 0.1, tau_confidence: float = 0.9,
           eig_min: float = 0.0) -> Action:
    """Route a decision.

    - ACT: epistemic uncertainty < eps AND recalibrated confidence >= tau.
    - QUERY MORE: epistemic uncertainty is high (reducible) and the expected
      information gain is worth it (eig >= eig_min, or eig unknown).
    - DEFER: aleatoric-dominant / irreducible (low epistemic but low confidence),
      a not-worth-querying high-epistemic case, or a flagged-sensitive topic.

    `confidence` should be the RECALIBRATED probability (see engine.calibration),
    never raw verbalized confidence.
    """
    if sensitive:
        return Action.DEFER
    if epistemic_std < eps_epistemic:
        # low model uncertainty: act if confident, else the doubt is irreducible -> defer
        return Action.ACT if confidence >= tau_confidence else Action.DEFER
    # high epistemic uncertainty: reducible by more data
    if eig is None or eig >= eig_min:
        return Action.QUERY_MORE
    return Action.DEFER
