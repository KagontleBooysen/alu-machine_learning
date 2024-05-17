#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class NST:
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize NST class with style and content images, alpha and beta."""
        if not self._is_valid_image(style_image):
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')
        if not self._is_valid_image(content_image):
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def _is_valid_image(image):
        """Check if the image is a valid numpy array with shape (h, w, 3)."""
        return isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3

    @staticmethod
    def scale_image(image):
        """Scale and preprocess the image."""
        if not NST._is_valid_image(image):
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')
        max_dims = 512
        shape = image.shape[:2]
        scale = max_dims / max(shape[0], shape[1])
        new_shape = (int(scale * shape[0]), int(scale * shape[1]))
        image = np.expand_dims(image, axis=0)
        image = tf.clip_by_value(tf.image.resize(image, new_shape, method='bicubic') / 255.0, 0.0, 1.0)
        return image

    def load_model(self):
        """Load the VGG19 model and modify it for NST."""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        x = vgg.input
        model_outputs = []
        content_output = None
        for layer in vgg.layers[1:]:
            if "pool" in layer.name:
                x = tf.keras.layers.AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides, name=layer.name)(x)
            else:
                x = layer(x)
                if layer.name in self.style_layers:
                    model_outputs.append(x)
                if layer.name == self.content_layer:
                    content_output = x
                layer.trainable = False
        model_outputs.append(content_output)
        self.model = tf.keras.models.Model(vgg.input, model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Compute the Gram matrix for a given input layer."""
        if not (isinstance(input_layer, tf.Tensor) or isinstance(input_layer, tf.Variable)) or input_layer.shape.ndims != 4:
            raise TypeError('input_layer must be a tensor of rank 4')
        _, nh, nw, _ = input_layer.shape.dims
        G = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        return G / tf.cast(nh * nw, tf.float32)

    def generate_features(self):
        """Generate the style and content features."""
        preprocessed_s = tf.keras.applications.vgg19.preprocess_input(self.style_image * 255)
        preprocessed_c = tf.keras.applications.vgg19.preprocess_input(self.content_image * 255)
        style_features = self.model(preprocessed_s)[:-1]
        self.content_feature = self.model(preprocessed_c)[-1]
        self.gram_style_features = [self.gram_matrix(style_feature) for style_feature in style_features]

    def layer_style_cost(self, style_output, gram_target):
        """Compute the style cost for a single layer."""
        if not (isinstance(style_output, tf.Tensor) or isinstance(style_output, tf.Variable)) or style_output.shape.ndims != 4:
            raise TypeError('style_output must be a tensor of rank 4')
        gram_style = self.gram_matrix(style_output)
        return tf.reduce_sum(tf.square(gram_style - gram_target)) / tf.square(tf.cast(gram_style.shape[-1], tf.float32))

    def style_cost(self, style_outputs):
        """Compute the overall style cost."""
        if not isinstance(style_outputs, list) or len(style_outputs) != len(self.style_layers):
            raise TypeError(f'style_outputs must be a list with a length of {len(self.style_layers)}')
        J_style = tf.add_n([self.layer_style_cost(style_outputs[i], self.gram_style_features[i]) for i in range(len(style_outputs))])
        J_style /= tf.cast(len(style_outputs), tf.float32)
        return J_style

