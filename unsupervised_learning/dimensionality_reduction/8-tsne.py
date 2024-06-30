#!/usr/bin/env python3

"""TSNE implementation."""

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Compute the t-sne dimension reduction
    :param X: Containing the dataset to be transformed by t-SNE
    :param ndims: New dimensional representation of X
    :param idims: Intermediate dimensional representation of X after PCA
    :param perplexity: Is the perplexity
    :param iterations: Is the number of iterations
    :param lr: Is the learning rate
    :return: Y containing the optimized low dimensional transformation of X
    """
    momentum_coeff = 0.8
    n, d = X.shape

    # Perform PCA on the input data
    pca_res = pca(X, idims)
    
    # Compute P affinities
    P = P_affinities(pca_res, perplexity=perplexity)
    P *= 4  # Early exaggeration

    # Initialize the low-dimensional representation
    Y = np.random.randn(n, ndims)
    Y_m1 = np.copy(Y)
    
    for i in range(iterations):
        # Compute gradients
        dY, Q = grads(Y, P)
        
        # Update Y using the gradients
        Y_new = Y - lr * dY + momentum_coeff * (Y - Y_m1)
        Y_new = Y_new - np.mean(Y_new, axis=0)
        
        # Update momentum
        Y_m1 = np.copy(Y)
        Y = np.copy(Y_new)

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

    return Y
