#!/usr/bin/env python3
"""
NST - Initialize
"""
import numpy as np
import tensorflow as tf

class NST:
    """class used to perform tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """define and initialize variables"""
        
        err_1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(err_1)
        if style_image.ndim != 3 or style_image.shape[-1] != 3:
            raise TypeError(err_1)
        err_2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(content_image, np.ndarray):
            raise TypeError(err_2)
        if content_image.ndim != 3 or content_image.shape[-1] != 3:
            raise TypeError(err_2)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Preprocessed style image
        self.style_image = self.scale_image(style_image)
        # Preprocessed content image
        self.content_image = self.scale_image(content_image)
        # Weight for content cost
        self.alpha = alpha
        # Weight for style cost
        self.beta = beta

        # Load the VGG19 model for the cost calculation
        self.load_model()

        # Instantiate the gram_style_features and content_feature attributes
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        function that rescales an image such that its pixels values are
        between 0 and 1 and its largest side is 512 pixels
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

        image = tf.image.resize(image, new_shape)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image[tf.newaxis, :]

    def load_model(self):
        """function that instantiates a VGG19 model from Keras"""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """function that calculates a Gram matrix, taking a layer as input"""
        err = "input_layer must be a tensor of rank 4"
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError(err)
        if len(input_layer.shape) != 4:
            raise TypeError(err)

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        result = result / num_locations
        return result

    def generate_features(self):
        """function that extracts the style and content features used to calculate the neural style cost"""
        style_image = tf.keras.applications.vgg19.preprocess_input(self.style_image * 255.0)
        content_image = tf.keras.applications.vgg19.preprocess_input(self.content_image * 255.0)

        style_outputs = self.model(style_image)[:-1]
        content_output = self.model(content_image)[-1]

        self.gram_style_features = [self.gram_matrix(style_output) for style_output in style_outputs]
        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """function that calculates the style cost for a single style_output layer"""
        c = style_output.shape[-1]
        err_1 = "style_output must be a tensor of rank 4"
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError(err_1)
        if len(style_output.shape) != 4:
            raise TypeError(err_1)
        err_2 = ("gram_target must be a tensor of shape [1, {}, {}]".format(c, c))
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(err_2)
        if gram_target.shape != (1, c, c):
            raise TypeError(err_2)

        gram_style = self.gram_matrix(style_output)
        style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))
        return style_cost

    def style_cost(self, style_outputs):
        """function that calculates the style cost for all style_output layers"""
        err = "style_outputs must be a list with a length of {}".format(len(self.style_layers))
        if not isinstance(style_outputs, list):
            raise TypeError(err)
        if len(self.style_layers) != len(style_outputs):
            raise TypeError(err)

        style_costs = []
        weight = 1 / len(self.style_layers)

        for style_output, gram_target in zip(style_outputs, self.gram_style_features):
            layer_style_cost = self.layer_style_cost(style_output, gram_target)
            weighted_layer_style_cost = weight * layer_style_cost
            style_costs.append(weighted_layer_style_cost)

        style_cost = tf.add_n(style_costs)
        return style_cost

# Example usage:
if __name__ == "__main__":
    style_image = np.random.rand(224, 224, 3).astype(np.float32)
    content_image = np.random.rand(224, 224, 3).astype(np.float32)
    nst = NST(style_image, content_image)
    print(nst.style_cost(nst.gram_style_features))

