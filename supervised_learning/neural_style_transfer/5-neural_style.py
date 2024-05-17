#!/usr/bin/env python3

"""
This module implements Neural Style Transfer (NST) using TensorFlow and VGG19.
NST is used to combine the content of one image with the style of another.
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    NST class performs Neural Style Transfer.

    Attributes:
        style_layers (list): List of VGG19 layer names used.
        content_layer (str): VGG19 layer name used for content extraction.
        style_image (np.ndarray): The style image.
        content_image (np.ndarray): The content image.
        alpha (float): Weight for the content cost.
        beta (float): Weight for the style cost.
        model (tf.keras.Model): The VGG19 model modified.
        content_feature (tf.Tensor): The content feature.
        gram_style_features (list): List of Gram matrices for the style image.
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST instance.

        Args:
            style_image (np.ndarray): The style image.
            content_image (np.ndarray): The content image.
            alpha (float): Weight for the content cost.
            beta (float): Weight for the style cost.

        Raises:
            TypeError: If style_image or content_image
                       with shape (h, w, 3) or if alpha/beta.
        """
        tf.enable_eager_execution()
        if (type(style_image) is not np.ndarray or style_image.ndim != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if (type(content_image) is not np.ndarray or content_image.ndim != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if (type(alpha) is not int and type(alpha) is not float) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if (type(beta) is not int and type(beta) is not float) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its largest dimension is 512 pixels.

        Args:
            image (np.ndarray): The image to be scaled.

        Returns:
            np.ndarray: The scaled image.

        Raises:
            TypeError: If the input image is not a valid numpy shape (h, w, 3).
        """
        if (type(image) is not np.ndarray or image.ndim != 3 or
                image.shape[2] != 3):
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')
        max_dims = 512
        shape = image.shape[:2]
        scale = max_dims / max(shape[0], shape[1])
        new_shape = (int(scale * shape[0]), int(scale * shape[1]))
        image = np.expand_dims(image, axis=0)
        image = tf.clip_by_value(
            tf.image.resize_bicubic(image, new_shape) / 255.0, 0.0, 1.0)
        return image

    def load_model(self):
        """
        Loads the VGG19 model and modifies it to output intermediate layers.
        """
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights='imagenet')
        x = vgg.input
        model_outputs = []
        content_output = None
        for layer in vgg.layers[1:]:
            if "pool" in layer.name:
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    name=layer.name
                )(x)
            else:
                x = layer(x)
                if layer.name in self.style_layers:
                    model_outputs.append(x)
                if layer.name == self.content_layer:
                    content_output = x
                layer.trainable = False
        model_outputs.append(content_output)
        model = tf.keras.models.Model(vgg.input, model_outputs)
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Computes the Gram matrix of an input tensor.

        Args:
            input_layer (tf.Tensor or tf.Variable): The input tensor.

        Returns:
            tf.Tensor: The Gram matrix.

        Raises:
            TypeError: If the input_layer is not a tensor of rank 4.
        """
        if not (isinstance(input_layer, tf.Tensor) or
                isinstance(input_layer, tf.Variable)) or \
                input_layer.shape.ndims != 4:
            raise TypeError('input_layer must be a tensor of rank 4')
        _, nh, nw, _ = input_layer.shape.dims
        G = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        return G / tf.cast(nh * nw, tf.float32)

    def generate_features(self):
        """
        Extracts the features used to compute the style and content cost.
        """
        preprocessed_s = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        preprocessed_c = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        style_features = self.model(preprocessed_s)[:-1]
        self.content_feature = self.model(preprocessed_c)[-1]
        self.gram_style_features = [
            self.gram_matrix(style_feature)
            for style_feature in style_features
        ]

    def layer_style_cost(self, style_output, gram_target):
        """
        Computes the style cost for a single layer.

        Args:
            style_output (tf.Tensor or tf.Variable): The style output tensor.
            gram_target (tf.Tensor or tf.Variable): The target Gram matrix.

        Returns:
            tf.Tensor: The style cost for the layer.

        Raises:
            TypeError: If style_output is not a tensor of rank 4 or
                       gram_target is not a tensor of the correct shape.
        """
        if not (isinstance(style_output, tf.Tensor) or
                isinstance(style_output, tf.Variable)) or \
                style_output.shape.ndims != 4:
            raise TypeError('style_output must be a tensor of rank 4')
        m, _, _, nc = style_output.shape.dims
        if not (isinstance(gram_target, tf.Tensor) or
                isinstance(gram_target, tf.Variable)) or \
                gram_target.shape.dims != [m, nc, nc]:
            raise TypeError(
                'gram_target must be a tensor of shape [{}, {}, {}]'.format(
                    m, nc, nc))
        gram_style = self.gram_matrix(style_output)
        return tf.reduce_sum(
            tf.square(gram_style - gram_target)) / tf.square(
            tf.cast(nc, tf.float32))

    def style_cost(self, style_outputs):
        """
        Computes the total style cost from all the style layers.

        Args:
            style_outputs (list of tf.Tensor or tf.Variable): The style output.

        Returns:
            tf.Tensor: The total style cost.

        Raises:
            TypeError: If style_outputs is not a list with the correct length.
        """
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
