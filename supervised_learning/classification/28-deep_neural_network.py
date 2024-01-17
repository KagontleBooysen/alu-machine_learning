#!/usr/bin/env python3
"""Deep neural network"""
import pickle
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class"""
    def __init__(self, nx, layers, activation='sig'):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__weights["W1"] = np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
        self.__weights["b1"] = np.zeros([layers[0], 1], dtype=float)
        if layers[0] <= 0 or type(layers[0]) is not int:
            raise TypeError("layers must be a list of positive integers")
        for i in range(1, self.L):
            if layers[i] <= 0 or type(layers[i]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(
                i + 1)] = np.random.randn(layers[i],
                                          layers[i-1])*np.sqrt(2/layers[i-1])
            self.__weights["b{}".format(
                i + 1)] = np.zeros([layers[i], 1], dtype=float)

    @property
    def L(self):
        return(self.__L)

    @property
    def cache(self):
        return(self.__cache)

    @property
    def weights(self):
        return(self.__weights)

    @property
    def activation(self):
        return(self.__activation)

    def forward_prop(self, X):
        """Forward propag multi layer"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            Z = np.dot(self.__weights["W{}".format(i)],
                       self.__cache["A{}".format(
                        i - 1)]) + self.__weights["b{}".format(i)]
            if i != self.__L:
                if self.__activation == 'sig':
                    a = 1 / (1 + np.exp(-Z))
                else:
                    a = np.sinh(Z) / np.cosh(Z)
            else:
                t = np.exp(Z)
                a = np.exp(Z) / np.sum(t, axis=0, keepdims=True)
            self.__cache["A{}".format(i)] = a
        return (a, self.__cache)

    def cost(self, Y, A):
        """Calculate cost"""
        m = Y.shape[1]
        m__loss = -(np.sum(Y * np.log(A), axis=1))
        cost = np.sum(m__loss) / m
        return (cost)

    def evaluate(self, X, Y):
        """Evaluate deep neural network"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A == np.amax(A, axis=0), 1, 0)
        return(prediction, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network"""
        A2 = cache["A{}".format(self.__L)]
        dz = A2 - Y
        for i in range(self.__L, 0, -1):
            db = (np.sum(dz, axis=1, keepdims=True) / Y.shape[1])
            dw = (np.matmul(cache["A{}".format(i - 1)], dz.T) / Y.shape[1])
            if self.__activation == "sig":
                dz = np.matmul(self.__weights["W{}".format(
                    i)].T, dz) * (cache["A{}".format(
                        i - 1)] * (1 - cache["A{}".format(i - 1)]))
            elif self.__activation == "tanh":
                dz = np.matmul(self.__weights["W{}".format(
                    i)].T, dz) * (1 - cache["A{}".format(i - 1)]
                                  * cache["A{}".format(i - 1)])
            self.__weights["b{}".format(i)] = self.__weights["b{}".format(
                i)] - (alpha * db)
            self.__weights["W{}".format(i)] = self.__weights["W{}".format(
                i)] - (alpha * dw).T

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            x_points = np.arange(0, iterations + 1, step)
            points = []
        for itr in range(iterations):
            A, cache = self.forward_prop(X)
            if verbose and (itr % step) == 0:
                cost = self.cost(Y, A)
                print("Cost after " + str(itr) + " iterations: " + str(cost))
            if graph and (itr % step) == 0:
                cost = self.cost(Y, A)
                points.append(cost)
            self.gradient_descent(Y, cache, alpha)
        itr += 1
        if verbose:
            cost = self.cost(Y, A)
            print("Cost after " + str(itr) + " iterations: " + str(cost))
        if graph:
            cost = self.cost(Y, A)
            points.append(cost)
            y_points = np.asarray(points)
            plt.plot(x_points, y_points, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return (self.evaluate(X, Y))

    def save(self, filename):
        """Save file object"""
        if filename[-4:] != '.pkl':
            filename = filename + '.pkl'
        fileobj = open(filename, 'wb')
        pickle.dump(self, fileobj)

    def load(filename):
        """Load file object"""
        try:
            fileobj = open(filename, 'rb')
            return pickle.load(fileobj)
        except(OSError, IOError) as e:
            return None
