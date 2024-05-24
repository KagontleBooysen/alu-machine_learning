#!/usr/bin/env python3
""" LSTM Cell """
import numpy as np


class LSTMCell:
    """ Represents an LSTM unit """
    def __init__(self, i, h, o):
        """ Class constructor

        - i is the dimensionality of the data
        - h is the dimensionality of the hidden state
        - o is the dimensionality of the outputs
        - Public instance attributes Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by
          that represent the weights and biases of the cell
            - Wf and bf are for the forget gate
            - Wu and bu are for the update gate
            - Wc and bc are for the intermediate cell state
            - Wo and bo are for the output gate
            - Wy and by are for the outputs
        - The weights should be initialized using a random normal distribution
          in the order listed above
        - The weights will be used on the right side for matrix multiplication
        - The biases should be initialized as zeros
        """
        self.Wf = np.random.normal(0, 1, (i + h, h))
        self.Wu = np.random.normal(0, 1, (i + h, h))
        self.Wc = np.random.normal(0, 1, (i + h, h))
        self.Wo = np.random.normal(0, 1, (i + h, h))
        self.Wy = np.random.normal(0, 1, (h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """ Sigmoid function """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """ Softmax function """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """ performs forward propagation for one time step

            - x_t is a numpy.ndarray of shape (m, i) that contains the data
              input for the cell
                - m is the batch size for the data
            - h_prev is a numpy.ndarray of shape (m, h) containing the previous
              hidden state
            - c_prev is a numpy.ndarray of shape (m, h) containing the previous
              cell state
            - The output of the cell should use a softmax activation function
            Returns: h_next, c_next, y
                - h_next is the next hidden state
                - c_next is the next cell state
                - y is the output of the cell
        """
        h = np.concatenate((h_prev, x_t), axis=1)

        f_t = self.sigmoid(np.dot(h, self.Wf) + self.bf)
        i_t = self.sigmoid(np.dot(h, self.Wu) + self.bu)
        c_t = np.tanh(np.dot(h, self.Wc) + self.bc)
        o_t = self.sigmoid(np.dot(h, self.Wo) + self.bo)
        c = f_t * c_prev + i_t * c_t
        h_next = o_t * np.tanh(c)
        y = np.dot(h_next, self.Wy) + self.by

        return h_next, c, self.softmax(y)
