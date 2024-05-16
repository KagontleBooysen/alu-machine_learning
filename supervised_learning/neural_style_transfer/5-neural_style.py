#!/usr/bin/env python3
"""Neural Style Transfer module"""
import tensorflow as tf
import numpy as np


class NST:
    """Neural Style Transfer class

    Attributes:
        style_layers (list): The pretrained selected layers for the style.
        content_layer (string): Represents the pretrained selected layer for
            the content.

    Raises:
        TypeError: If style_image is not np.ndarray with shape (h, w, 3)
        TypeError: If content_image is not np.ndarray with shape (h, w, 3)
        TypeError: If alpha is a negative value.
        TypeError: If beta is a negative value.

    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializer"""
        if not isinstance(style_image, np.ndarray) \
                or len(style_image.shape) != 3 \
                or style_image.shape[2] != 3:
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )
        if not isinstance(content_image, np.ndarray) \
                or len(content_image.shape) != 3 \
                or content_image.shape[2] != 3:
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()
        self.style_image = NST.scale_image(style_image)
        self.content_image = NST.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales the image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image (np.ndarray): Contains the image to be scaled of shape
                (h, w, 3) where respectively the height the width and the
                number of channels.

        Returns:
            tf.Tensor: The scaled image

        """
        if not isinstance(image, np.ndarray) \
                or len(image.shape) != 3 \
                or image.shape[2] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )
        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * h_new / h)
        else:
            w_new = 512
            h_new = int(h * w_new / w)

        image = image[tf.newaxis, ...]
        image = tf.image.resize_bicubic(
            image,
            [h_new, w_new]
        )
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        return image

    def load_model(self):
        """Creates the model used to calculate the cost

        """
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            weights='imagenet'
        )
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg.save('source_model')
        custom_model = tf.keras.models.load_model(
            'source_model',
            custom_objects=custom_objects
        )

        custom_model.trainable = False
        for layer in custom_model.layers:
            layer.trainable = False

        s_outputs = [
            custom_model.get_layer(name).output for name in self.style_layers
        ]
        c_output = custom_model.get_layer(self.content_layer).output
        outputs = s_outputs + [c_output]
        self.model = tf.keras.models.Model(custom_model.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrices

        Args:
            input_layer (tf.Tensor): Contains the layer output whose gram
                matrix should be calculated of shape (1, h, w, c)

        Returns:
            tf.Tensor: Contains the gram matrix of shape (1, c, c)

        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) \
                or tf.rank(input_layer).numpy() != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        _, h, w, c = tf.shape(input_layer).numpy()
        F = tf.reshape(input_layer, (1, -1, c))
        n = tf.shape(F)[1]
        gram = tf.matmul(F, F, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """Extracts the features used to calculate neural style cost

        """
        nb_layers = len(self.style_layers)

        style_img = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        content_img = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_img)
        content_outputs = self.model(content_img)

        style_features = [
            layer for layer in style_outputs[:nb_layers]
        ]
        self.content_feature = content_outputs[nb_layers:][0]

        self.gram_style_features = [
            NST.gram_matrix(layer) for layer in style_features
        ]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer

        Args:
            style_output: Contains the layer style output of the generated
                image.
            gram_target: Gram matrix of the target style output for that layer

        Returns:
            Layer's style cost

        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) \
                or tf.rank(style_output).numpy() != 4:
            raise TypeError('style_output must be a tensor of rank 4')

        _, h, w, c = tf.shape(style_output).numpy()
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) \
                or gram_target.shape != (1, c, c):
            raise TypeError(
                'gram_target must be a tensor of shape [1, {}, {}]'
                .format(c, c)
            )
        gram_style = NST.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculates the style cost for generated image

        Args:
            style_outputs (list(tf.Tensor)): style outputs for the generated
                image.

        Returns:
            The style cost

        """
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) \
                or len(style_outputs) != length + 1:
            raise TypeError('style_outputs must be

