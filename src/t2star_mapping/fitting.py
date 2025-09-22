from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares


FitMethod = Literal["ols", "gls", "nlls", "num"]


@dataclass
class FitResult:
	T2star_ms: float
	S0: float
	Sfit: np.ndarray
	r_squared: float
	iterations: int
	converged: bool


def _r_squared(y: np.ndarray, yhat: np.ndarray) -> float:
	ss_res = np.sum((y - yhat) ** 2)
	ss_tot = (y.size - 1) * np.var(y)
	if ss_tot == 0:
		return 0.0
	return float(1.0 - ss_res / ss_tot)


def _ols_gls_common(S: np.ndarray, TE_ms: np.ndarray, X: np.ndarray, weights: np.ndarray | None = None) -> Tuple[float, float]:
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
	T2s = -1.0 / beta[0]
	S0 = float(np.exp(beta[1]))
	return float(T2s), S0


def fit_t2star(S: ArrayLike, TE_ms: ArrayLike, method: FitMethod = "nlls") -> FitResult:
	S = np.asarray(S, dtype=float).ravel()
	TE_ms = np.asarray(TE_ms, dtype=float).ravel()
	assert S.size == TE_ms.size
	nt = S.size
	X = np.c_[TE_ms, np.ones(nt)]

	if method == "ols":
		T2s, S0 = _ols_gls_common(S, TE_ms, X, None)
		Sfit = S0 * np.exp(-TE_ms / T2s) if np.isfinite(T2s) else np.full_like(S, np.nan)
		return FitResult(T2star_ms=float(T2s), S0=float(S0), Sfit=Sfit, r_squared=_r_squared(S, Sfit), iterations=1, converged=np.isfinite(T2s))

	if method == "gls":
		mask = S > 0
		weights = np.zeros_like(S)
		weights[mask] = 1.0 / S[mask]
		T2s, S0 = _ols_gls_common(S, TE_ms, X, weights)
		Sfit = S0 * np.exp(-TE_ms / T2s) if np.isfinite(T2s) else np.full_like(S, np.nan)
		return FitResult(T2star_ms=float(T2s), S0=float(S0), Sfit=Sfit, r_squared=_r_squared(S, Sfit), iterations=1, converged=np.isfinite(T2s))

	if method == "num":
		if nt < 2:
			return FitResult(T2star_ms=np.nan, S0=np.nan, Sfit=np.full_like(S, np.nan), r_squared=0.0, iterations=1, converged=False)
		T2s = (TE_ms[-1] - TE_ms[0]) * (S[0] + S[-1] + 2 * np.sum(S[1:-1])) / (2 * (nt - 1) * (S[0] - S[-1]))
		S0 = S[0] * np.exp(TE_ms[0] / T2s)
		Sfit = S0 * np.exp(-TE_ms / T2s)
		return FitResult(T2star_ms=float(T2s), S0=float(S0), Sfit=Sfit, r_squared=_r_squared(S, Sfit), iterations=1, converged=True)

	# NLLS (default): initialize with GLS
	T2s0, S00 = _ols_gls_common(S, TE_ms, X, np.where(S > 0, 1.0 / S, 0.0))
	if not np.isfinite(T2s0) or T2s0 <= 0 or not np.isfinite(S00) or S00 <= 0:
		# fallback to OLS init
		T2s0, S00 = _ols_gls_common(S, TE_ms, X, None)
		if not np.isfinite(T2s0) or T2s0 <= 0:
			return FitResult(T2star_ms=np.nan, S0=np.nan, Sfit=np.full_like(S, np.nan), r_squared=0.0, iterations=1, converged=False)

	def model(p: np.ndarray, te: np.ndarray) -> np.ndarray:
		return p[0] * np.exp(-te / p[1])

	def resid(p: np.ndarray) -> np.ndarray:
		return model(p, TE_ms) - S

	p0 = np.array([S00, T2s0], dtype=float)
	bounds = (np.array([0.0, 1e-3]), np.array([np.inf, 1e5]))
	res = least_squares(resid, p0, bounds=bounds, max_nfev=200)
	S0 = float(res.x[0])
	T2s = float(res.x[1])
	Sfit = model(res.x, TE_ms)
	return FitResult(T2star_ms=T2s, S0=S0, Sfit=Sfit, r_squared=_r_squared(S, Sfit), iterations=int(res.nfev), converged=bool(res.success))
