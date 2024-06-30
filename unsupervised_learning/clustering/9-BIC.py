#!/usr/bin/env python3

"""Module to calculate BIC for various k values using the Expectation-Maximization algorithm."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Calculates BIC over various values of k.

    Parameters:
    X : numpy.ndarray
        The data set to analyze.
    kmin : int, optional
        The minimum number of clusters to check for (default is 1).
    kmax : int, optional
        The maximum number of clusters to check for (default is None, which means n).
    iterations : int, optional
        The maximum number of iterations for the EM algorithm (default is 1000).
    tol : float, optional
        The tolerance for the EM algorithm (default is 1e-5).
    verbose : bool, optional
        If True, print information about the EM algorithm (default is False).

    Returns:
    best_k : int
        The best number of clusters.
    best_result : tuple
        The parameters (pi, m, S) for the best number of clusters.
    log_likelihood : numpy.ndarray
        The log likelihood for each value of k.
    b : numpy.ndarray
        The BIC value for each value of k.
    """
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
            X, k, iterations=iterations, tol=tol, verbose=verbose)
        results.append((pi, m, S))
        p = k * (d + 2) * (d + 1) / 2 - 1
        b[k - kmin] = p * np.log(n) - 2 * log_likelihood[k - kmin]

    amin = np.argmin(b)
    best_k = amin + kmin
    best_result = results[amin]

    return best_k, best_result, log_likelihood, b
