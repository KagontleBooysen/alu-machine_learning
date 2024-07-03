#!/usr/bin/env python3
"""
3-forward.py
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    function that performs the forward algorithm for a hidden markov model
    """

    # Initial: shape (N, 1), N: number of hidden states
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    if not np.isclose(np.sum(Initial, axis=0), [1])[0]:
        return None, None
    # Transition: shape (N, N)
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None
    if not np.isclose(np.sum(Transition, axis=1), np.ones(Initial.shape[0])).all():
        return None, None
    # Observation: shape (T,), T: number of observations
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    # Emission: shape (N, M), M: number of all possible observations
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not np.isclose(np.sum(Emission, axis=1), np.ones(Emission.shape[0])).all():
        return None, None

    # N: Number of hidden states
    N = Initial.shape[0]
    # T: Number of observations
    T = Observation.shape[0]

    # Initialize F (equivalent to alpha): shape (N, T)
    F = np.zeros((N, T))
    # Initial step
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Iterate over the observations
    for j in range(1, T):
        for i in range(N):
            F[i, j] = np.sum(Emission[i, Observation[j]] * Transition[:, i] * F[:, j - 1], axis=0)

    # Likelihood of the observations
    P = np.sum(F[:, T - 1])

    return P, F
