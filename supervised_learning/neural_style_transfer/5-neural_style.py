#!/usr/bin/env python3
"""
NST - Neural Style Transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """Class used to perform tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize the NST class with style and content images"""
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()
        self.gram_style_features = self.generate_features(self.style_image)
        self.content_feature = self.generate_features(self.content_image)[-1]

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.
        """
        err = "image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(image, np.ndarray):
            raise TypeError(err)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise TypeError(err)

        max_dim = 512
        long_dim = max(image.shape[:-1])
        scale = max_dim / long_dim
        new_shape = tuple(map(lambda x: int(scale * x), image.shape[:-1]))
        image = np.expand_dims(image, axis=0)
        image = tf.image.resize(image, new_shape)
        image = image / 255.0
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

        return image

    def load_model(self):
        """Instantiates a VGG19 model from Keras"""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)
        model = tf.keras.models.Model([vgg.input], outputs)
        return model

    @staticmethod
    def gram_matrix(input_tensor):
        """
        Calculates the Gram matrix for a given input tensor.
        """
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def generate_features(self, image):
        """
        Extracts the style and content features from the given image.
        """
        preprocessed_image = tf.keras.applications.vgg19.preprocess_input(
            image * 255)
        outputs = self.model(preprocessed_image)
        return outputs

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for the generated image.
        """
        if not isinstance(style_outputs, list) or len(style_outputs) != len(self.style_layers):
            raise TypeError(f'style_outputs must be a list with a length of {len(self.style_layers)}')

        style_cost = 0
        weight_per_style_layer = 1.0 / float(len(self.style_layers))

        for output, gram_target in zip(style_outputs, self.gram_style_features):
            gram_style = self.gram_matrix(output)
            current_style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))
            style_cost += weight_per_style_layer * current_style_cost

        return style_cost

