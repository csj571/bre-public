"""From-scratch Gaussian Process regression (spec components 1/2/4).

Exact GP posterior via Cholesky factorization with a recursive-jitter guard for
ill-conditioned kernels; hyperparameters (ell, eta, sigma_n) fit by maximizing
the log-marginal likelihood with L-BFGS-B in log-space. `predict()` returns the
posterior mean AND the epistemic std; `get_bald_score()` turns that std into
the BALD / mutual-information acquisition score used for active learning.
numpy + scipy.optimize only — no sklearn, no torch.
"""
import numpy as np
import scipy.optimize


class GaussianProcessRegressor:
    """Exact GP regressor over kernels rbf / matern32 / matern52 and the
    cosine-distance variants (cosine_rbf, cosine_matern52) that the study
    frontend's gp.js mirrors. Stateful: fit() stores the Cholesky factor and
    weights; predict() before fit() returns the prior (zero mean, eta std)."""

    def __init__(self, kernel_type='matern52', ell=1.5, eta=1.0, sigma_n=0.2):
        self.kernel_type = kernel_type
        self.ell = ell
        self.eta = eta
        self.sigma_n = sigma_n
        
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None
        
    def _kernel(self, X1, X2):
        """
        Computes the covariance matrix between X1 (N, D) and X2 (M, D).
        """
        # Distance squared: d2(x, y) = ||x||^2 + ||y||^2 - 2 x^T y
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True).T
        d2 = np.clip(X1_sq + X2_sq - 2.0 * np.dot(X1, X2.T), 0, None)
        d = np.sqrt(d2)
        
        if self.kernel_type == 'rbf':
            return (self.eta**2) * np.exp(-0.5 * d2 / (self.ell**2))
        elif self.kernel_type == 'matern32':
            s = np.sqrt(3.0) * d / self.ell
            return (self.eta**2) * (1.0 + s) * np.exp(-s)
        elif self.kernel_type == 'matern52':
            s = np.sqrt(5.0) * d / self.ell
            return (self.eta**2) * (1.0 + s + (s**2) / 3.0) * np.exp(-s)
        elif self.kernel_type == 'cosine_rbf':
            norms1 = np.linalg.norm(X1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(X2, axis=1, keepdims=True).T
            dot = np.dot(X1, X2.T)
            cos_sim = dot / (np.dot(norms1, norms2) + 1e-8)
            d_cos = 1.0 - cos_sim
            return (self.eta**2) * np.exp(-0.5 * (d_cos**2) / (self.ell**2))
        elif self.kernel_type == 'cosine_matern52':
            norms1 = np.linalg.norm(X1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(X2, axis=1, keepdims=True).T
            dot = np.dot(X1, X2.T)
            cos_sim = dot / (np.dot(norms1, norms2) + 1e-8)
            d_cos = 1.0 - cos_sim
            s = np.sqrt(5.0) * d_cos / self.ell
            return (self.eta**2) * (1.0 + s + (s**2) / 3.0) * np.exp(-s)
        else:
            return (self.eta**2) * np.exp(-0.5 * d2 / (self.ell**2))
            
    def fit(self, X, y):
        """
        Fits the GP model on training features X (N, D) and targets y (N,).
        """
        self.X = X
        self.y = y
        N = X.shape[0]

        K_base = self._kernel(X, X)

        # Recursive jitter to ensure positive-definiteness. Clear any factor
        # from a previous fit first, so a refit that exhausts every jitter
        # level raises below instead of silently keeping stale state.
        self.L = None
        self.alpha = None
        jitter = 1e-6
        max_jitter = 1e-1

        while jitter < max_jitter:
            try:
                K = K_base + (self.sigma_n**2 + jitter) * np.eye(N)
                self.L = np.linalg.cholesky(K)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
                
        if self.L is None:
            raise np.linalg.LinAlgError("Cholesky decomposition failed even with max jitter.")
        
        # Solve L L^T alpha = y => L w = y, L^T alpha = w
        w = np.linalg.solve(self.L, y)
        self.alpha = np.linalg.solve(self.L.T, w)
        
    def optimize(self):
        """
        Optimizes hyperparameters (ell, eta, sigma_n) by maximizing the log-marginal likelihood.
        """
        if self.X is None or self.y is None:
            raise ValueError("Must fit() model before optimizing hyperparameters.")
            
        def neg_log_marginal_likelihood(params):
            # params = [log(ell), log(eta), log(sigma_n)]
            ell_prop = np.exp(params[0])
            eta_prop = np.exp(params[1])
            sigma_n_prop = np.exp(params[2])
            
            # Temporarily set params
            orig_ell, orig_eta, orig_sigma_n = self.ell, self.eta, self.sigma_n
            self.ell, self.eta, self.sigma_n = ell_prop, eta_prop, sigma_n_prop
            
            N = self.X.shape[0]
            K_base = self._kernel(self.X, self.X)
            
            # Attempt decomposition
            jitter = 1e-6
            L_temp = None
            while jitter < 1e-1:
                try:
                    K = K_base + (sigma_n_prop**2 + jitter) * np.eye(N)
                    L_temp = np.linalg.cholesky(K)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
                    
            # If still fails, return a huge penalty
            if L_temp is None:
                self.ell, self.eta, self.sigma_n = orig_ell, orig_eta, orig_sigma_n
                return 1e10
                
            w = np.linalg.solve(L_temp, self.y)
            alpha_temp = np.linalg.solve(L_temp.T, w)
            
            # nlml = 0.5 * y^T alpha + sum(log(diag(L))) + 0.5 * N * log(2*pi)
            nlml = 0.5 * np.dot(self.y, alpha_temp) + np.sum(np.log(np.diag(L_temp))) + 0.5 * N * np.log(2.0 * np.pi)
            
            # Restore original params
            self.ell, self.eta, self.sigma_n = orig_ell, orig_eta, orig_sigma_n
            return nlml

        # Initial guess (log-transformed to enforce positivity)
        init_params = np.array([np.log(self.ell), np.log(self.eta), np.log(self.sigma_n)])
        
        # Optimize
        res = scipy.optimize.minimize(
            neg_log_marginal_likelihood, 
            init_params, 
            method='L-BFGS-B',
            bounds=[(np.log(1e-2), np.log(1e2)), (np.log(1e-2), np.log(1e2)), (np.log(1e-4), np.log(1e1))]
        )
        
        if res.success:
            self.ell = np.exp(res.x[0])
            self.eta = np.exp(res.x[1])
            self.sigma_n = np.exp(res.x[2])
            
            # Refit with optimal params
            self.fit(self.X, self.y)
        return res.success
        
    def predict(self, Xstar):
        """
        Predicts mean and epistemic std at test points Xstar (M, D).
        """
        M = Xstar.shape[0]
        if self.X is None:
            # Prior prediction
            return np.zeros(M), np.ones(M) * self.eta
            
        K_star = self._kernel(Xstar, self.X)  # (M, N)
        
        # Expected value: K_star @ alpha
        mu = np.dot(K_star, self.alpha)
        
        # Variance: K(X*, X*) - K_star @ K_obs^-1 @ K_star^T
        # Solve L v = K_star^T
        v = np.linalg.solve(self.L, K_star.T)  # (N, M)
        kss = self.eta**2
        varF = kss - np.sum(v**2, axis=0)      # (M,)
        varF = np.maximum(varF, 1e-12)
        sigmaF = np.sqrt(varF)
        
        return mu, sigmaF
        
    def get_bald_score(self, sigmaF, sigma_n=None):
        """
        Computes the mutual information (epistemic entropy / BALD score) for a given sigmaF.
        Supports scalar or vector-based heteroscedastic noise (sigma_n).
        """
        if sigma_n is None:
            sigma_n = self.sigma_n
        return 0.5 * np.log(1.0 + (sigmaF**2) / (sigma_n**2 + 1e-8))
