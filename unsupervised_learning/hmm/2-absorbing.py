#!/usr/bin/env python3
"""
Determines if a markov chain is absorbing
"""
import numpy as np

def absorbing(P):
    """
    Determines if a markov chain is absorbing
    :param P: square 2D numpy.ndarray of shape (n, n) representing the
    transition matrix
    :return: True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray:
        return False
    if len(P.shape) != 2:
        return False
    n, n_t = P.shape
    if n != n_t:
        return False
    sum_test = np.sum(P, axis=1)
    for elem in sum_test:
        if not np.isclose(elem, 1):
            return False

    # Check if there is at least one absorbing state
    diagonal = np.diag(P)
    absorbing_states = (diagonal == 1)
    if not np.any(absorbing_states):
        return False

    # Create a reachability matrix
    reachability = np.linalg.matrix_power(P, n)
    
    # Check if every state can reach at least one absorbing state
    for i in range(n):
        if not np.any(reachability[i, absorbing_states]):
            return False

    return True

# Test the function with the provided matrix
matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.5, 0.5], [0, 0.5, 0.5, 0]])
print(absorbing(matrix))  # Expected output: True

