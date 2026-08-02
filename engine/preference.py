"""Online Bayesian preference learning over a linear reward model.

`BayesianPreferenceLearner.update_posterior_laplace(phi_w, phi_l)` consumes
preferred/dispreferred feature pairs (Bradley-Terry likelihood) and performs a
Laplace approximation: Adam finds the MAP weight vector, the Hessian at the MAP
gives the posterior precision (Cholesky-inverted, with a jittered fallback for
ill-conditioned cases). The posterior (mu, cov) is kept in module buffers and
folded into the prior on the next call, so updates are sequential.
`get_reward_distribution()` returns reward mean/std for LCB-style objectives —
the point of carrying a posterior at all is that a downstream policy can be
penalized for acting on rewards it is not confident about.

This is the ONLY engine module that requires torch (the `bre-engine[torch]`
extra); everything else in engine/ stays numpy/stdlib. Upstream this module is
named `somatic_bayesian.SomaticBayesianEngine` — see PROVENANCE.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor(nn.Module):
    """Reference feature map from token sequences to the linear reward model's
    feature space: embeddings into a GRU, final hidden state as the feature
    vector. Any encoder producing a fixed-width vector works — this one is here
    so the learner has a runnable end-to-end example on sequence inputs."""

    def __init__(self, vocab_size=10, embed_dim=16, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x: shape (batch, seq_len)
        embeds = self.embedding(x)
        _, hidden = self.gru(embeds)  # hidden: shape (1, batch, hidden_dim)
        return hidden.squeeze(0)      # shape (batch, hidden_dim)


class BayesianPreferenceLearner(nn.Module):
    """Bayesian linear reward model with sequential Laplace posterior updates
    (see module docstring). Posterior state lives in registered buffers so it
    serializes with the module but is never touched by an optimizer."""

    def __init__(self, feature_dim=16, init_var=1.0):
        super().__init__()
        self.feature_dim = feature_dim

        # Register buffers so they are part of module state but not PyTorch parameters
        self.register_buffer('mu', torch.zeros(feature_dim))
        self.register_buffer('cov', torch.eye(feature_dim) * init_var)
        self.register_buffer('prior_mu', torch.zeros(feature_dim))
        self.register_buffer('prior_prec', torch.eye(feature_dim) / init_var)
        self.init_var = init_var

    def get_reward_distribution(self, phi):
        """
        Computes the mean and standard deviation of reward predictions.
        phi: shape (batch, feature_dim)
        """
        # Expected reward: mu^T * phi
        mean = torch.matmul(phi, self.mu)  # shape (batch,)

        # Variance: phi^T * cov * phi
        # var[i] = phi[i] @ cov @ phi[i]^T
        var = torch.sum(torch.matmul(phi, self.cov) * phi, dim=-1)  # shape (batch,)
        std = torch.sqrt(torch.clamp(var, min=1e-6))

        return mean, std

    def update_posterior_laplace(self, phi_w, phi_l, steps=25, lr=0.01):
        """
        Updates the posterior mean (mu) and covariance (cov) using the Laplace approximation
        based on new preference pairs (phi_w is preferred, phi_l is dispreferred).
        phi_w: shape (N, feature_dim)
        phi_l: shape (N, feature_dim)
        """
        # Detach inputs from the computation graph to prevent multi-backward autograd errors
        phi_w = phi_w.detach()
        phi_l = phi_l.detach()

        # Feature differences: delta_phi = phi_w - phi_l
        delta_phi = phi_w - phi_l
        N = delta_phi.shape[0]

        prior_prec = self.prior_prec

        # Optimize mu to find the MAP (Maximum A Posteriori) estimate
        # Log posterior: log p(w) + sum log p(y_w > y_l | w)
        # = -0.5 * (w - prior_mu)^T @ prior_prec @ (w - prior_mu) + sum log sigmoid(w^T @ delta_phi)
        with torch.enable_grad():
            current_mu = self.mu.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([current_mu], lr=lr)

            for _ in range(steps):
                optimizer.zero_grad()

                # Log likelihood term
                logits = torch.matmul(delta_phi, current_mu)
                log_lik = F.logsigmoid(logits).sum()

                # Prior term
                diff = current_mu - self.prior_mu
                prior_term = -0.5 * torch.dot(diff, torch.matmul(prior_prec, diff))

                # Objective: minimize negative log posterior
                loss = -(log_lik + prior_term)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            self.mu.copy_(current_mu)

            # Compute posterior precision using the Hessian at the MAP estimate
            # Prec = Prior_Prec + sum p_i * (1 - p_i) * delta_phi_i * delta_phi_i^T
            # where p_i = sigmoid(mu^T @ delta_phi_i)
            logits = torch.matmul(delta_phi, self.mu)
            p = torch.sigmoid(logits)
            weights = p * (1.0 - p)  # shape (N,)

            # Vectorized Hessian: X^T W X
            W_delta_phi = weights.unsqueeze(1) * delta_phi
            hessian_lik = torch.matmul(delta_phi.T, W_delta_phi)

            prec = prior_prec + hessian_lik

            # To get new_cov, solve prec * new_cov = I
            # using Cholesky for stability:
            try:
                L = torch.linalg.cholesky(prec)
                new_cov = torch.cholesky_solve(torch.eye(self.feature_dim, device=self.mu.device), L)
            except torch.linalg.LinAlgError:
                # Fallback with jitter if ill-conditioned
                prec_jittered = prec + torch.eye(self.feature_dim, device=self.mu.device) * 1e-4
                new_cov = torch.linalg.solve(prec_jittered, torch.eye(self.feature_dim, device=self.mu.device))

            self.cov.copy_(new_cov)

            # Update the prior parameters for future updates (sequential updates)
            self.prior_mu.copy_(self.mu)
            self.prior_prec.copy_(prec)

        logger.debug(
            "Laplace posterior update complete. Mean norm: %.4f, Cov trace: %.4f",
            self.mu.norm().item(), self.cov.trace().item(),
        )
