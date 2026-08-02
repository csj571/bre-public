"""Adaptive 1-D Kalman filter (BUILD_PLAN B5) — Python port of the JS
`AdaptiveKalman` in bre1-simulator/signal.js.

Tracks a scalar latent value from noisy measurements. The observation-noise
estimate `r` adapts to the EWMA of squared innovations, so the filter widens
when the stream gets noisier. Pure Python (no numpy needed)."""


class AdaptiveKalman:
    def __init__(self, q: float = 0.01, r: float = 0.5, ewma: float = 0.1):
        self._q0, self._r0, self._ewma0 = q, r, ewma
        self.q = q              # process noise (fixed)
        self.r = r              # observation noise (adapts)
        self.ewma = ewma        # EWMA rate for residual variance
        self.x = 0.0
        self.P = 1.0
        self.resid_var = r
        self.initialized = False

    def update(self, z: float) -> dict:
        z = float(z)
        if not self.initialized:
            self.x = z
            self.initialized = True
            return {"filtered": self.x, "P": self.P, "r": self.r, "innovation": 0.0}
        # Predict
        x_pred = self.x
        P_pred = self.P + self.q
        # Update
        innov = z - x_pred
        S = P_pred + self.r
        K = P_pred / S
        self.x = x_pred + K * innov
        self.P = (1.0 - K) * P_pred
        # Adapt observation noise from the EWMA of squared innovations
        self.resid_var = (1.0 - self.ewma) * self.resid_var + self.ewma * innov * innov
        self.r = 0.7 * self.r + 0.3 * self.resid_var
        return {"filtered": self.x, "P": self.P, "r": self.r, "innovation": innov}

    def reset(self) -> None:
        self.__init__(self._q0, self._r0, self._ewma0)


def kalman_filter(series, **kwargs):
    """Convenience: run an AdaptiveKalman over a 1-D series, return filtered values."""
    kf = AdaptiveKalman(**kwargs)
    return [kf.update(z)["filtered"] for z in series]
