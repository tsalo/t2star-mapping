from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares


FitMethod = Literal["ols", "gls", "nlls", "num"]


@dataclass
class FitResult:
    """Result of T2* fitting for a single voxel/series.

    Parameters
    ----------
    t2star_ms : float
        Estimated T2* time in milliseconds.
    s0 : float
        Estimated signal at TE=0 (extrapolated).
    s_fit : numpy.ndarray
        Modeled signal across the provided echo times.
    r_squared : float
        Coefficient of determination of the fit.
    iterations : int
        Number of optimizer evaluations/iterations used.
    converged : bool
        Whether the fitting procedure converged.
    """

    t2star_ms: float
    s0: float
    s_fit: np.ndarray
    r_squared: float
    iterations: int
    converged: bool


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
    S: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[float, float]:
    """Linearized T2* fit helper for OLS/GLS in log-domain.

    Parameters
    ----------
    S : numpy.ndarray
        Signal across echoes (shape: [T]). Must be strictly positive where used.
    X : numpy.ndarray
        Design matrix with columns [TE_ms, 1].
    weights : numpy.ndarray or None, optional
        If provided, diagonal weights for GLS (e.g., 1/S for heteroscedasticity).

    Returns
    -------
    t2star_ms : float
        Estimated T2* in ms, from slope of log-linear model.
    s0 : float
        Estimated s0 from intercept of log-linear model.
    """
    mask = S > 0
    if mask.sum() < 2:
        return np.nan, np.nan

    y = np.log(S[mask])
    Xm = X[mask]
    if weights is None:
        beta = np.linalg.pinv(Xm) @ y
    else:
        W = np.diag(weights[mask])
        beta = np.linalg.pinv(Xm.T @ W @ Xm) @ (Xm.T @ W @ y)

    t2s = -1.0 / beta[0]
    s0 = float(np.exp(beta[1]))
    return float(t2s), s0


def fit_t2star(S: ArrayLike, TE_ms: ArrayLike, method: FitMethod = "nlls") -> FitResult:
    """Estimate T2* from multi-echo signal using various methods.

    Parameters
    ----------
    S : array_like
        Signal across echoes (length T). Non-negative.
    TE_ms : array_like
        Echo times in milliseconds (length T). Must align with ``S`` order.
    method : {"ols", "gls", "nlls", "num"}, optional
        Fitting method. Default is "nlls".
        - "ols": ordinary least squares in log-domain.
        - "gls": generalized least squares in log-domain (weights ~ 1/S).
        - "nlls": non-linear least squares to s0*exp(-TE/T2*).
        - "num": numerical approximation (Hagberg-style).

    Returns
    -------
    FitResult
        Structured result with T2*, s0, fitted signal, R^2, iterations, and convergence.
    """
    S = np.asarray(S, dtype=float).ravel()
    TE_ms = np.asarray(TE_ms, dtype=float).ravel()
    assert S.size == TE_ms.size
    nt = S.size
    X = np.c_[TE_ms, np.ones(nt)]

    if method == "ols":
        T2s, s0 = _ols_gls_common(S, X, None)
        s_fit = (
            s0 * np.exp(-TE_ms / T2s) if np.isfinite(T2s) else np.full_like(S, np.nan)
        )
        return FitResult(
            t2star_ms=float(T2s),
            s0=float(s0),
            s_fit=s_fit,
            r_squared=_r_squared(S, s_fit),
            iterations=1,
            converged=np.isfinite(T2s),
        )

    if method == "gls":
        mask = S > 0
        weights = np.zeros_like(S)
        weights[mask] = 1.0 / S[mask]
        T2s, s0 = _ols_gls_common(S, X, weights)
        s_fit = (
            s0 * np.exp(-TE_ms / T2s) if np.isfinite(T2s) else np.full_like(S, np.nan)
        )
        return FitResult(
            t2star_ms=float(T2s),
            s0=float(s0),
            s_fit=s_fit,
            r_squared=_r_squared(S, s_fit),
            iterations=1,
            converged=np.isfinite(T2s),
        )

    if method == "num":
        if nt < 2:
            return FitResult(
                t2star_ms=np.nan,
                s0=np.nan,
                s_fit=np.full_like(S, np.nan),
                r_squared=0.0,
                iterations=1,
                converged=False,
            )
        T2s = (
            (TE_ms[-1] - TE_ms[0])
            * (S[0] + S[-1] + 2 * np.sum(S[1:-1]))
            / (2 * (nt - 1) * (S[0] - S[-1]))
        )
        s0 = S[0] * np.exp(TE_ms[0] / T2s)
        s_fit = s0 * np.exp(-TE_ms / T2s)
        return FitResult(
            t2star_ms=float(T2s),
            s0=float(s0),
            s_fit=s_fit,
            r_squared=_r_squared(S, s_fit),
            iterations=1,
            converged=True,
        )

    # NLLS (default): initialize with GLS
    T2s0, S00 = _ols_gls_common(S, X, np.where(S > 0, 1.0 / S, 0.0))
    if not np.isfinite(T2s0) or T2s0 <= 0 or not np.isfinite(S00) or S00 <= 0:
        # fallback to OLS init
        T2s0, S00 = _ols_gls_common(S, X, None)
        if not np.isfinite(T2s0) or T2s0 <= 0:
            return FitResult(
                t2star_ms=np.nan,
                s0=np.nan,
                s_fit=np.full_like(S, np.nan),
                r_squared=0.0,
                iterations=1,
                converged=False,
            )

    def model(p: np.ndarray, te: np.ndarray) -> np.ndarray:
        """Exponential model S(te) = s0 * exp(-te/T2*)."""
        return p[0] * np.exp(-te / p[1])

    def resid(p: np.ndarray) -> np.ndarray:
        """Residuals for least squares optimization."""
        return model(p, TE_ms) - S

    p0 = np.array([S00, T2s0], dtype=float)
    bounds = (np.array([0.0, 1e-3]), np.array([np.inf, 1e5]))
    res = least_squares(resid, p0, bounds=bounds, max_nfev=200)
    s0 = float(res.x[0])
    T2s = float(res.x[1])
    s_fit = model(res.x, TE_ms)
    return FitResult(
        t2star_ms=T2s,
        s0=s0,
        s_fit=s_fit,
        r_squared=_r_squared(S, s_fit),
        iterations=int(res.nfev),
        converged=bool(res.success),
    )
