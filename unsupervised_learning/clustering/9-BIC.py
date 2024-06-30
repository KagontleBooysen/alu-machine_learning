  1 #!/usr/bin/env python3
  2
  3
  4 """useless comment"""
  5
  6
  7 import numpy as np
  8 expectation_maximization = __import__('8-EM').expectation_maximization
  9
 10
 11 def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
 12     """calculates BIC over various """
 13     if type(X) is not np.ndarray or X.ndim != 2:
 14         return None, None, None, None
 15     n, d = X.shape
 16     if type(kmin) is not int or kmin != int(kmin) or kmin < 1:
 17         return None, None, None, None
 18     if kmax is None:
 19         kmax = n
 20     if type(kmax) is not int or kmax != int(kmax) or kmax < 1:
 21         return None, None, None, None
 22     if kmax <= kmin:
 23         return None, None, None, None
 24     if (type(iterations) is not int or iterations != int(iterations) or iterations < 1):
 25         return None, None, None, None
 26     if type(tol) is not float or tol < 0:
 27         return None, None, None, None
 28     if type(verbose) is not bool:
 29         return None, None, None, None
 30     b = np.zeros(kmax + 1 - kmin)
 31     log_likelihood = np.zeros(kmax + 1 - kmin)
 32     results = []
 33     for k in range(kmin, kmax + 1):
 34         pi, m, S, _, log_likelihood[k - kmin] = expectation_maximization(
 35                 X, k, iterations=iterations, tol=tol, verbose=verbose)
 36         results.append((pi, m, S))
 37         p = k * (d + 2) * (d + 1) / 2 - 1
 38         b[k - kmin] = p * np.log(n) - 2 * log_likelihood[k - kmin]
 39     amin = np.argmin(b)
 40     best_k = amin + kmin
 41     best_result = results[amin]
 42     return best_k, best_result, log_likelihood, b
