#!/usr/bin/env python3
""""""
import tensorflow as tf


def create_layer(prev, n):
    """ Create a layer and return output tensor of layer
        @prev: is the tensor output of the previous layer
        @n: is the number of nodes in the layer to create
        @activation: is the activation function that the layer should use
        Returns: the tensor output of the layer
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, name='layer', kernel_initializer=w)
    out = layer(prev)
    return out


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
        @Data_train: is a tuple containing the training inputs
                     and training labels, respectively
        @Data_valid: is a tuple containing the validation inputs
                     and validation labels, respectively
        @layers: is a list containing the number of nodes in each
                 layer of the network
        @activation: is a list containing the activation functions
                     used for each layer of the network
        @alpha: is the learning rate
        @beta1: is the weight for the first moment of Adam Optimization
        @beta2: is the weight for the second moment of Adam Optimization
        @epsilon: is a small number used to avoid division by zero
        @decay_rate: is the decay rate for inverse time decay of the
                     learning rate
                     (the corresponding decay step should be 1)
        @batch_size: is the number of data points that should be in a
                     mini-batch
        @epochs: is the number of times the training should pass through
                 the whole dataset
        @save_path: is the path where the model should be saved to
        Returns: the path where the model was saved
    """
    # Set training data
    X_train = Data_train[0]
    Y_train = Data_train[1]
    # Set validation data
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]
    x = tf.placeholder("float", X_train.shape, 'x')
    y = tf.placeholder("float", Y_train.shape, 'y')
    for i in range(len(layers)):
        out_layer = create_layer(x, layers[i])
    saver = tf.train.Saver()
    with tf.Session() as sess:
        return saver.save(sess, save_path)
