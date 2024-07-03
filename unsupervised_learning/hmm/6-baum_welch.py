#!/usr/bin/env python3
"""markov hidden model"""
import numpy as np


def initialize_alphas(X, initial, emission):
    """
    Initializes the forward variables in the hidden Markov model.
    alpha[j, t] = P(Z_t = j, X_1:X_t)
    """
    N, M = emission.shape
    T = X.shape[0]
    alpha = np.zeros((N, T))
    alpha[:, 0] = initial.flatten() * emission[:, X[0]]
    return alpha


def forward(Observation, Emission, Transition, Initial):
    """Forward algorithm for a hidden Markov model
    Forward algorithm recursive computation:
      alpha[j, t] = sum(alpha[i, t-1] * A[i, j]) * B[j, Observation[t]]
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape[0] != Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    alpha = initialize_alphas(Observation, Initial, Emission)

    for t in range(1, T):
        for j in range(N):
            alpha[j, t] = np.sum(
                alpha[:, t - 1] * Transition[:, j]) *\
                Emission[j, Observation[t]]

    P = np.sum(alpha[:, T - 1])
    return P, alpha


def initialize_betas(X, emission):
    """
    Initializes the backward variables in the hidden Markov model.
    beta[j, t] = P(X_{t+1}:X_T | Z_t = j)
    """
    N, M = emission.shape
    T = X.shape[0]
    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))
    return beta


def backward(Observation, Emission, Transition, Initial):
    """backward algorithm for a hidden markov model
    Backward algorithm recursive computation:
    beta[j, t] = sum(A[i, j] * B[j, Observation[t+1]] * beta[i, t+1])
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    if Transition.shape[0] != Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    beta = initialize_betas(Observation, Emission)

    for t in range(T - 2, -1, -1):
        for j in range(N):
            beta[j, t] = np.sum(
                beta[:, t + 1] * Transition[j, :] *
                Emission[:, Observation[t + 1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * beta[:, 0])
    return P, beta


def gamma_zeta(X, Emission, Transition, Initial):
    """
    Compute gamma and zeta values in Baum-Welch algorithm.
    gamma[i, t] = P(Z_t = i | X, model)
    zeta[i, j, t] = P(Z_t = i, Z_{t+1} = j | X, model)
    Zeta calculation for t:
    zeta[i, j, t] = alpha[i, t] * A[i, j] * B[j, X[t+1]] * beta[j, t+1]
    """
    _, alpha = forward(X, Emission, Transition, Initial)
    _, beta = backward(X, Emission, Transition, Initial)
    gamma = np.zeros((Emission.shape[0], X.shape[0]))
    zeta = np.zeros((Emission.shape[0], Emission.shape[0], X.shape[0] - 1))
    for t in range(X.shape[0]):
        gamma[:, t] = alpha[:, t] * beta[:, t] / \
            np.sum(alpha[:, t] * beta[:, t])
        if t != X.shape[0] - 1:
            for i in range(Emission.shape[0]):
                for j in range(Emission.shape[0]):
                    zeta[i, j, t] = alpha[i, t] * Transition[i, j] *\
                        Emission[j, X[t + 1]] * beta[j, t + 1]
            zeta[:, :, t] /= np.sum(zeta[:, :, t])
    return gamma, zeta


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """performs the Baum-Welch algorithm for a hidden markov model
    Update Transition matrix using zeta and gamma Formula:
    Transition[i, j] = sum(zeta[i, j, t]) / sum(gamma[i, t])
    (where t is from 0 to T-1)
    Update Emission matrix using gamma and observations Formula:
    Emission[i, k] = sum(gamma[i, t]
    where Observations[t] = k) / sum(gamma[i, t])
    (where t is from 0 to T-1)
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    N, M = Emission.shape
    T = Observations.shape[0]
    for n in range(iterations):
        gamma, zeta = gamma_zeta(Observations, Emission, Transition, Initial)
        Transition = np.sum(zeta, axis=2) / np.sum(gamma[:, :-1], axis=1)
        gamma_sum = np.sum(gamma, axis=1)
        gamma_sum = gamma_sum.reshape((-1, 1))
        for i in range(M):
            gamma_sum = gamma_sum.reshape((-1, 1))
            Emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)

        # Normalize Emission matrix
        Emission = Emission / gamma_sum

    return Transition, Emission
