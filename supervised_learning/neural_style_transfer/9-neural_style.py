#!/usr/bin/env python3

"""
This module contains a class NST with functions that
calculate style cost
"""
import numpy as np
import tensorflow as tf


class NST:
    """class NST"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """class constructor
        style_image - image used as style reference stored as numpy array
        content_image - image used as content reference stored as numpy array
        alpha - weight for content cost
        beta - weight for style cost
        model - the Keras model used to calculate cost
        """
        # tensorflow executes eagerly
        tf.enable_eager_execution()

        error = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(error)
        if style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(error)

        error = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(content_image, np.ndarray):
            raise TypeError(error)
        if content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(error)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        # load model
        self.load_model()

    @staticmethod
    def scale_image(image):
        """rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        image - numpy.ndarray of shape (h, w, 3) with image to be scaled
        Returns a scaled image with shape (h, w, 3) - max(h, w) is 512 pixels
        """
        err = "image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(image, np.ndarray):
            raise TypeError(err)
        if image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(err)

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * h_new / h)
        else:
            w_new = 512
            h_new = int(h * w_new / w)

        image = image[tf.newaxis, ...]
        image = tf.image.resize_bicubic(image, (h_new, w_new))
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        return image

    def load_model(self):
        """creates the model used to calculate cost
        model should output the style and content layers"""
        base_model = tf.keras.applications.VGG19(include_top=False,
                                                 weights='imagenet')
        base_model.save("vgg19_base_model")

        objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}

        vgg19 = tf.keras.models.load_model('vgg19_base_model',
                                           custom_objects=objects)

        for layers in vgg19.layers:
            layers.trainable = False

        style_outputs = [vgg19.get_layer(layer).output
                         for layer in self.style_layers]
        content_outputs = vgg19.get_layer(self.content_layer).output

        model_outputs = style_outputs + [content_outputs]

        model = tf.keras.models.Model(vgg19.input, model_outputs)
        self.model = model
        # return model

    def gram_matrix(input_layer):
        """
        input_layer - has layer output whose gram matrix
        should be calculated
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if input_layer.ndim != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        # F - features
        F = tf.reshape(input_layer, (h * w, c))
        # n = tf.shape(F)[0]
        n = int(h * w)
        gram = tf.matmul(F, F, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        gram /= tf.cast(n, tf.float32)
        return (gram)

    def generate_features(self):
        """
        extracts the features used to calculate neural style cost
        """
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)

        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        outputs_style = self.model(style_image)

        style_outputs = outputs_style[:-1]

        outputs_content = self.model(content_image)

        content_ouput = outputs_content[-1]

        self.gram_style_features = [self.gram_matrix(style_output)
                                    for style_output in style_outputs]

        self.content_feature = content_ouput

    def layer_style_cost(self, style_output, gram_target):
        """calculate style cost for single layer"""

        c = style_output.shape[-1]
        err_1 = "style_output must be a tensor of rank 4"
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError(err_1)
        if len(style_output.shape) != 4:
            raise TypeError(err_1)
        err_2 = ("gram_target must be a tensor of shape [1, {}, {}]".
                 format(c, c))
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(err_2)
        if gram_target.shape != (1, c, c):
            raise TypeError(err_2)

        # Compute the gram matrix of the style_output layer
        gram_style = self.gram_matrix(style_output)
        # Calculate the mean squared error between gram_style and gram_target
        style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return style_cost

    def style_cost(self, style_outputs):
        """calculate style cost for generated image"""

        if not isinstance(style_outputs, list) or \
                len(style_outputs) != len(self.style_layers):
            raise TypeError(
                'style_outputs must be a list with a length of {}'.format(
                    len(self.style_layers)))
        J_style = tf.add_n([
            self.layer_style_cost(
                style_outputs[i], self.gram_style_features[i])
            for i in range(len(style_outputs))
        ])
        J_style /= tf.cast(len(style_outputs), tf.float32)
        return J_style

    def content_cost(self, content_output):
        """style cost for generated image"""
        if not (isinstance(content_output, tf.Tensor) or
                isinstance(content_output, tf.Variable)) or \
                content_output.shape.dims != self.content_feature.shape.dims:
            raise TypeError('content_output must be a tensor of shape {}'
                            .format(self.content_feature.shape))
        _, nh, nw, nc = content_output.shape.dims
        return tf.reduce_sum(tf.square(content_output -
                                       self.content_feature)) / \
            tf.cast(nh * nw * nc, tf.float32)

    def total_cost(self, generated_image):
        """total cost"""
        if not (isinstance(generated_image, tf.Tensor) or
                isinstance(generated_image, tf.Variable)) or \
                generated_image.shape.dims != self.content_image.shape.dims:
            raise TypeError('generated_image must be a tensor of shape {}'
                            .format(self.content_image.shape))
        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        model_outputs = self.model(preprocessed)
        style_outputs = [style_layer for style_layer in model_outputs[:-1]]
        content_output = model_outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J = (self.alpha * J_content) + (self.beta * J_style)
        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """calculate gradients for generated image"""
        err = "generated_image must be a tensor of shape {}".format(
            self.content_image.shape)
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(err)
        if generated_image.shape != self.content_image.shape:
            raise TypeError(err)

        with tf.GradientTape() as tape:
            loss = self.total_cost(generated_image)
            total_cost, content_cost, style_cost = loss

        gradients = tape.gradient(total_cost, generated_image)

        return (gradients, total_cost, content_cost, style_cost)


    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """generate the neutral style transfered image"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be positive")
        if step is not None and type(step) is not int:
            raise TypeError("step must be an integer")
        if step is not None and (step < 0 or step > iterations):
            raise ValueError("step must be positive and less than iterations")
        if type(lr) is not int and type(lr) is not float:
            raise TypeError("lr must be a number")
        if lr < 0:
            raise ValueError("lr must be positive")
        if type(beta1) is not float:
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if type(beta2) is not float:
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")
        generated_image = self.content_image
        cost = 0
        return generated_image, cost
