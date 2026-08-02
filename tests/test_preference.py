"""BayesianPreferenceLearner unit tests.

These require torch — the `bre-engine[torch]` extra — and the whole module skips
cleanly when it isn't installed, so the numpy-only test suite stays green on a
bare environment.
"""
import sys
import os

import pytest

torch = pytest.importorskip("torch")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from engine.preference import BayesianPreferenceLearner, FeatureExtractor  # noqa: E402


def test_laplace_update_moves_the_mean_and_shrinks_the_covariance():
    feature_dim = 8
    learner = BayesianPreferenceLearner(feature_dim=feature_dim)

    # Generate mock preference pairs
    phi_w = torch.randn((5, feature_dim))
    phi_l = torch.randn((5, feature_dim))

    prior_trace = learner.cov.trace().item()

    # Run online Laplace update
    learner.update_posterior_laplace(phi_w, phi_l, steps=5, lr=0.01)

    # Verify posterior parameters changed (mean is non-zero, covariance trace shrinks)
    assert float(learner.mu.norm()) > 0.0
    assert learner.cov.trace().item() < prior_trace


def test_ill_conditioned_input_takes_the_jittered_fallback():
    feature_dim = 4
    learner = BayesianPreferenceLearner(feature_dim=feature_dim)

    # Highly correlated / identical inputs to cause ill-conditioned precision matrix
    phi_w = torch.ones((10, feature_dim))
    phi_l = torch.ones((10, feature_dim)) * 0.99999

    # This should trigger the Cholesky fallback and not crash
    learner.update_posterior_laplace(phi_w, phi_l, steps=5)

    assert not torch.isnan(learner.mu).any()
    assert not torch.isnan(learner.cov).any()


def test_sequential_updates_keep_tightening_the_posterior():
    """Each update folds the posterior into the prior, so evidence accumulates
    rather than being re-learned from scratch."""
    torch.manual_seed(0)
    learner = BayesianPreferenceLearner(feature_dim=6)

    traces = [learner.cov.trace().item()]
    for _ in range(3):
        phi_w = torch.randn((4, 6))
        phi_l = torch.randn((4, 6))
        learner.update_posterior_laplace(phi_w, phi_l, steps=5)
        traces.append(learner.cov.trace().item())

    assert traces == sorted(traces, reverse=True), traces


def test_reward_distribution_reports_uncertainty_per_point():
    """The reason for carrying a posterior at all: a downstream objective can
    penalize rewards the model is not confident about (LCB = mean - alpha*std)."""
    learner = BayesianPreferenceLearner(feature_dim=4, init_var=1.0)
    phi = torch.eye(4)

    mean, std = learner.get_reward_distribution(phi)
    assert mean.shape == (4,) and std.shape == (4,)
    assert torch.allclose(mean, torch.zeros(4))          # prior mean is zero
    assert torch.allclose(std, torch.ones(4), atol=1e-5)  # prior var is init_var

    # A direction the learner has seen should end up less uncertain than one it
    # has not. Feed evidence only along the first basis vector.
    phi_w = torch.zeros((8, 4))
    phi_w[:, 0] = 1.0
    learner.update_posterior_laplace(phi_w, torch.zeros((8, 4)), steps=10)

    _, std_after = learner.get_reward_distribution(phi)
    assert std_after[0] < std[0]
    assert torch.allclose(std_after[1:], std[1:], atol=1e-5)


def test_feature_extractor_maps_sequences_to_fixed_width_vectors():
    extractor = FeatureExtractor(vocab_size=10, embed_dim=8, hidden_dim=5)
    tokens = torch.randint(0, 10, (3, 7))       # batch of 3, sequence length 7
    assert extractor(tokens).shape == (3, 5)
