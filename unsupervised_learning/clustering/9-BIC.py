#!/usr/bin/env python3

"""Calculates BIC."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Calculates BIC for various cluster counts."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < 1:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    b = np.zeros(kmax + 1 - kmin)
    log_likelihood = np.zeros(kmax + 1 - kmin)
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, _, log_likelihood[k - kmin] = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )
        results.append((pi, m, S))
        p = k * (d + 2) * (d + 1) / 2 - 1
        b[k - kmin] = p * np.log(n) - 2 * log_likelihood[k - kmin]

    amin = np.argmin(b)
    best_k = amin + kmin
    best_result = results[amin]
    return best_k, best_result, log_likelihood, b
