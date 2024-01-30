#!/usr/bin/env python3
"""
A function  that updates the weights of a neural network with Dropout
regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    A function that upates the weights of a neural network with Dropout
    regularization using gradient descent"""
    A2 = cache["A{}".format(L)]
    dz = A2 - Y
    for i in range(L, 0, -1):
        db = (np.sum(dz, axis=1, keepdims=True) / Y.shape[1])
        dw = (np.matmul(cache["A{}".format(i - 1)], dz.T) / Y.shape[1])
        if (i - 1) > 0:
            dz = np.matmul(weights["W{}".format(
                i)].T, dz) * (1 - cache["A{}".format(i - 1)]
                              * cache["A{}".format(i - 1)])
            dz = dz * cache["D{}".format(i - 1)]
            dz = dz / keep_prob
        weights["b{}".format(i)] = weights["b{}".format(
            i)] - (alpha * db)
        weights["W{}".format(i)] = weights["W{}".format(
            i)] - (alpha * dw).T
