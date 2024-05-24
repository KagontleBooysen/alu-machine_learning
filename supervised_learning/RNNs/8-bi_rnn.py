#!/usr/bin/env python3
""" Bidirectional RNN """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ performs forward propagation for a bidirectional RNN

        - bi_cell is an instance of BidirectionalCell that will
          be used for the forward propagation
        - X is the data to be used, given as a numpy.ndarray of shape
          (t, m, i)
            - t is the maximum number of time steps
            - m is the batch size
            - i is the dimensionality of the data
        - h_0 is the initial hidden state in the forward direction, given
          as a numpy.ndarray of shape (m, h)
            - h is the dimensionality of the hidden state
        - h_t is the initial hidden state in the backward direction, given
          as a numpy.ndarray of shape (m, h)
        Returns: H, Y
            - H is a numpy.ndarray containing all of the concatenated
              hidden states
            - Y is a numpy.ndarray containing all of the outputs
    """
    T, m, i = X.shape
    _, h = h_0.shape
    H_f = np.zeros((T + 1, m, h))
    H_b = np.zeros((T + 1, m, h))
    H_f[0] = h_0
    H_b[-1] = h_t
    for f, b in zip(range(T), range(T - 1, -1, -1)):
        H_f[f + 1] = bi_cell.forward(H_f[f], X[f])
        H_b[b] = bi_cell.backward(H_b[b + 1], X[b])
    H = np.concatenate((H_f[1:], H_b[0:-1]), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
