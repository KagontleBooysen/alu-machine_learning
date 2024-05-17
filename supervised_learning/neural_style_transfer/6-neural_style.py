#!/usr/bin/env python3
"""
module which contains the class NST
"""
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf


class NST:
    """
    class which performs the neural style transfer technique
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor:
        * style_image - the image used as a style reference,
          stored as a numpy.ndarray
        * content_image - the image used as a content reference,
          stored as a numpy.ndarray
        * alpha - the weight for content cost
        * beta - the weight for style cost
        * Sets Tensorflow to execute eagerly
        * Sets the instance attributes:
            - style_image: the preprocessed style image
            - content_image: the preprocessed content image
            - alpha: the weight for content cost
            - beta: the weight for style cost
        """
        error = 'must be a numpy.ndarray with shape (h, w, 3)'

        ndim = style_image.ndim
        shape = style_image.shape[2]
        if type(style_image) != np.ndarray or ndim != 3 or shape != 3:
            raise TypeError('style_image {}'.format(error))

        ndim = content_image.ndim
        shape = content_image.shape[2]
        if type(content_image) != np.ndarray or ndim != 3 or shape != 3:
            raise TypeError('content_image {}'.format(error))

        if (type(alpha) != int and type(alpha) != float) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if (type(beta) != int and type(beta) != float) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()
        tf.enable_eager_execution()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
        * image - a numpy.ndarray of shape (h, w, 3) containing
          the image to be scaled
        * The scaled image should be a tf.tensor with the shape
          (1, h_new, w_new, 3) where max(h_new, w_new) == 512 and
          min(h_new, w_new) is scaled proportionately
        * The image should be resized using bicubic interpolation
        * After resizing, the image’s pixel values should be rescaled
          from the range [0, 255] to [0, 1].
        Returns: the scaled image
        """
        error = 'image must be a numpy.ndarray with shape (h, w, 3)'
        ndim = image.ndim
        shape = image.shape[2]
        if type(image) != np.ndarray or ndim != 3 or shape != 3:
            raise TypeError("{}".format(error))

        # calculating rescaling
        h, w, _ = image.shape
        max_dim = 512
        maximum = max(h, w)
        scale = max_dim / maximum
        new_shape = (int(h * scale), int(w * scale))
        image = np.expand_dims(image, axis=0)
        scaled_image = tf.image.resize_bicubic(image, new_shape)
        scaled_image = tf.clip_by_value(scaled_image / 255, 0, 1)

        return scaled_image

    def load_model(self):
        """
        * creates the model used to calculate cost
        * the model should use the VGG19 Keras model as a base
        * the model’s input should be the same as the VGG19 input
        * the model’s output should be a list containing the outputs of
          the VGG19 layers listed in style_layers followed by content _layer
        * saves the model in the instance attribute model
        """
        # for more information you can check in this blog (the link is sliced)
        # pt 1 https://medium.com/tensorflow/neural-style-transfer-creating-
        # pt 2 art-with-deep-learning-using-tf-keras-and-eager-
        # pt 3 execution-7d541ac31398
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        x = vgg.input

        style_outputs = []
        content_output = None

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size, strides=layer.strides,
                    padding=layer.padding)
                x = layer(x)
            else:
                x = layer(x)
                if layer.name in self.style_layers:
                    style_outputs.append(layer.output)

                if layer.name == self.content_layer:
                    content_output = x
                layer.trainable = False

        outputs = style_outputs + [content_output]

        return tf.keras.models.Model(vgg.input, outputs)


    @staticmethod
    def gram_matrix(input_layer):
        """
        * input_layer - an instance of tf.Tensor or tf.Variable of 
                        shape (1, h, w, c) containing the layer output
                        whose gram matrix should be calculated
        * if input_layer is not an instance of tf.Tensor or tf.Variable of
          rank 4, raise a TypeError with the message input_layer
          must be a tensor of rank 4
        Returns: a tf.Tensor of shape (1, c, c) containing
                 the gram matrix of input_layer
        """
        if input_layer.shape.ndims != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        if not isinstance(input_layer, tf.Tensor):
            raise TypeError("input_layer must be a tensor of rank 4")
        if isinstance(input_layer, tf.Variable):
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, _ = input_layer.shape

        # for more context visite
        # https://www.tensorflow.org/tutorials/generative/style_transfer
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        num_locations = tf.cast(h * w, tf.float32)
        
        return result / num_locations

    def generate_features(self):
        """
        extracts the features used to calculate neural style cost
        * Sets the public instance attributes:
            gram_style_features - a list of gram matrices calculated from
                                  the style layer outputs of the style image
            content_feature - the content layer output of the content image
        """
        s = self.style_image * 255
        p = self.content_image * 255
        prepro_style = tf.keras.applications.vgg19.preprocess_input(s)
        prepro_content = tf.keras.applications.vgg19.preprocess_input(p)

        style_ft = self.model(prepro_style)[:-1]

        self.gram_style_features = [self.gram_matrix(i) for i in style_ft]
        self.content_feature = self.model(prepro_content)[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        * style_output - tf.Tensor of shape (1, h, w, c) containing
                         the layer style output of the generated image
        * gram_target - tf.Tensor of shape (1, c, c) the gram matrix
                        of the target style output for that layer
        Returns: the layer’s style cost
        """

        if not isinstance(style_output, tf.Tensor):
            raise TypeError('style_output must be a tensor of rank 4')
        if isinstance(style_output, tf.Variable):
            raise TypeError('style_output must be a tensor of rank 4')
        if style_output.shape.ndims != 4:
            raise TypeError('style_output must be a tensor of rank 4')

        c = style_output.shape[-1]
        error = 'gram_target must be a tensor '
        error += 'of shape [1, {}, {}]'.format(c, c)
        if not isinstance(gram_target, tf.Tensor):
            raise TypeError(error)
        if isinstance(gram_target, tf.Variable):
            raise TypeError(error)
        if gram_target.shape.dims != [1, c, c]:
            raise TypeError(error)

        gram_style = self.gram_matrix(style_output)
  
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for generated image
        * style_outputs - a list of tf.Tensor style outputs
                          for the generated image
        each layer should be weighted evenly with all weights summing to 1
        Returns: the style cost
        """
        l = len(self.style_layers)
        error = 'style_outputs must be a list with a length of {}'.format(l)
        if len(style_outputs) != l:
            raise TypeError(error)

        style_score = 0
        # Accumulate style losses from all layers
        # Here, we equally weight each contribution of each loss layer
        weight_p_style_l = 1.0 / float(l)
        style_score = 0
        for style, features in zip(style_outputs, self.gram_style_features):
            l_style_cost = self.layer_style_cost(style, features)
            style_score += l_style_cost * weight_p_style_l

        return style_score

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image
        * content_output - a tf.Tensor containing the content output
          for the generated image
        Returns: the content cost
        """
        s = self.content_feature.shape
        error = 'content_output must be a tensor of shape {}'.format(s)
        if not isinstance(content_output, tf.Tensor):
            raise TypeError(error)
        if isinstance(content_output, tf.Variable):
            raise TypeError(error)
        if content_output.shape != s:
            raise TypeError(error)

        square = tf.square(content_output - self.content_feature)
        return tf.reduce_mean(square)

