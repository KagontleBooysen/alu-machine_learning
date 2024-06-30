#!/usr/bin/env python3

"""TSNE implementation with PCA and gradient descent."""

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Compute the t-sne dimension reduction.
    :param X: np.ndarray, shape (n_samples, n_features)
        The dataset to be transformed by t-SNE.
    :param ndims: int, optional, default=2
        New dimensional representation of X.
    :param idims: int, optional, default=50
        Intermediate dimensional representation of X after PCA.
    :param perplexity: float, optional, default=30.0
        The perplexity parameter for t-SNE.
    :param iterations: int, optional, default=1000
        Number of iterations for optimization.
    :param lr: float, optional, default=500
        Learning rate for gradient descent.
    :return: np.ndarray, shape (n_samples, ndims)
        The optimized low-dimensional representation of X.
    """
    momentum_coeff = 0.8
    n, d = X.shape

    # Apply PCA to reduce dimensionality
    pca_res = pca(X, idims)
    
    # Compute pairwise affinities
    P = P_affinities(pca_res, perplexity=perplexity)
    P *= 4  # Early exaggeration

    # Initialize the low-dimensional representation
    Y = []
    y = np.random.randn(n, ndims)
    Y.append(y)
    Y.append(y.copy())

    for i in range(iterations):
        # Compute gradients and current affinities
        dY, Q = grads(Y[-1], P)
        
        # Update Y using gradient descent with momentum
        y = Y[-1] - lr * dY + momentum_coeff * (Y[-1] - Y[-2])
        y = y - np.mean(y, axis=0)
        Y.append(y)

        # Print cost at every 100 iterations
        if (i + 1) % 100 == 0:
            current_cost = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, current_cost))

        # Adjust momentum after 20 iterations
        if i == 20:
            momentum_coeff = 0.5

        # End early exaggeration after 100 iterations
        if (i + 1) == 100:
            P /= 4

    return Y[-1]


# Example usage (assuming you have the necessary modules and data):
# X = np.random.randn(100, 50)  # Example data
# Y = tsne(X)
# print(Y)

