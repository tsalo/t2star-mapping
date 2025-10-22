from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares


FitMethod = Literal["ols", "gls", "nlls", "num"]


def _r_squared(y: np.ndarray, yhat: np.ndarray) -> float:
    """Compute the coefficient of determination R^2.

    Parameters
    ----------
    y : numpy.ndarray
        Observed values.
    yhat : numpy.ndarray
        Predicted values from a model.

    Returns
    -------
    float
        R^2 statistic, clipped to 0 if variance is zero.
    """
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = (y.size - 1) * np.var(y)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _ols_gls_common(
    s: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[float, float]:
    """Linearized T2* fit helper for OLS/GLS in log-domain.

    Parameters
    ----------
    s : numpy.ndarray
        Signal across echoes (shape: [T]). Must be strictly positive where used.
    x : numpy.ndarray
        Design matrix with columns [TE_ms, 1].
    weights : numpy.ndarray or None, optional
        If provided, diagonal weights for GLS (e.g., 1/s for heteroscedasticity).

    Returns
    -------
    t2star_ms : float
        Estimated T2* in ms, from slope of log-linear model.
    s0 : float
        Estimated s0 from intercept of log-linear model.
    """
    mask = s > 0
    if mask.sum() < 2:
        return np.nan, np.nan

    y = np.log(s[mask])
    xm = x[mask]
    if weights is None:
        beta = np.linalg.pinv(xm) @ y
    else:
        w = np.diag(weights[mask])
        beta = np.linalg.pinv(xm.T @ w @ xm) @ (xm.T @ w @ y)

    t2s = -1.0 / beta[0]
    s0 = float(np.exp(beta[1]))
    return float(t2s), s0


def fit_t2star(
    s: ArrayLike,
    tes_ms: ArrayLike,
    method: FitMethod = "nlls",
) -> Tuple[float, float, np.ndarray, float, int, bool]:
    """Estimate T2* from multi-echo signal using various methods.

    Parameters
    ----------
    s : array_like
        Signal across echoes (length T). Non-negative.
    tes_ms : array_like
        Echo times in milliseconds (length T). Must align with ``s`` order.
    method : {"ols", "gls", "nlls", "num"}, optional
        Fitting method. Default is "nlls".
        - "ols": ordinary least squares in log-domain.
        - "gls": generalized least squares in log-domain (weights ~ 1/s).
        - "nlls": non-linear least squares to s0*exp(-TE/T2*).
        - "num": numerical approximation (Hagberg-style).

    Returns
    -------
    t2star_ms : float
        Estimated T2* (ms).
    s0 : float
        Estimated s0.
    s_fit : numpy.ndarray
        Fitted signal across echoes.
    r_squared : float
        Coefficient of determination.
    iterations : int
        Number of optimizer evaluations/iterations used.
    converged : bool
        Whether the fitting procedure converged.
    """
    s = np.asarray(s, dtype=float).ravel()
    tes_ms = np.asarray(tes_ms, dtype=float).ravel()
    assert s.size == tes_ms.size
    nt = s.size
    x = np.c_[tes_ms, np.ones(nt)]

    if method == "ols":
        t2s, s0 = _ols_gls_common(s, x, None)
        s_fit = s0 * np.exp(-tes_ms / t2s) if np.isfinite(t2s) else np.full_like(s, np.nan)
        return float(t2s), float(s0), s_fit, _r_squared(s, s_fit), 1, np.isfinite(t2s)

    if method == "gls":
        mask = s > 0
        weights = np.zeros_like(s)
        weights[mask] = 1.0 / s[mask]
        t2s, s0 = _ols_gls_common(s, x, weights)
        s_fit = s0 * np.exp(-tes_ms / t2s) if np.isfinite(t2s) else np.full_like(s, np.nan)
        return float(t2s), float(s0), s_fit, _r_squared(s, s_fit), 1, np.isfinite(t2s)

    if method == "num":
        if nt < 2:
            return np.nan, np.nan, np.full_like(s, np.nan), 0.0, 1, False
        t2s = (tes_ms[-1] - tes_ms[0]) * (s[0] + s[-1] + 2 * np.sum(s[1:-1])) / (
            2 * (nt - 1) * (s[0] - s[-1])
        )
        s0 = s[0] * np.exp(tes_ms[0] / t2s)
        s_fit = s0 * np.exp(-tes_ms / t2s)
        return float(t2s), float(s0), s_fit, _r_squared(s, s_fit), 1, True

    # NLLS (default): initialize with GLS
    t2s0, s00 = _ols_gls_common(s, x, np.where(s > 0, 1.0 / s, 0.0))
    if not np.isfinite(t2s0) or t2s0 <= 0 or not np.isfinite(s00) or s00 <= 0:
        # fallback to OLS init
        t2s0, s00 = _ols_gls_common(s, x, None)
        if not np.isfinite(t2s0) or t2s0 <= 0:
            return np.nan, np.nan, np.full_like(s, np.nan), 0.0, 1, False

    def model(p: np.ndarray, te: np.ndarray) -> np.ndarray:
        """Exponential model S(te) = s0 * exp(-te/T2*)."""
        return p[0] * np.exp(-te / p[1])

    def resid(p: np.ndarray) -> np.ndarray:
        """Residuals for least squares optimization."""
        return model(p, tes_ms) - s

    p0 = np.array([s00, t2s0], dtype=float)
    bounds = (np.array([0.0, 1e-3]), np.array([np.inf, 1e5]))
    res = least_squares(resid, p0, bounds=bounds, max_nfev=200)
    s0 = float(res.x[0])
    t2s = float(res.x[1])
    s_fit = model(res.x, tes_ms)
    return float(t2s), float(s0), s_fit, _r_squared(s, s_fit), int(res.nfev), bool(res.success)
