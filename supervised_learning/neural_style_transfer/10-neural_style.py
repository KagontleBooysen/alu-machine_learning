#!/usr/bin/env python3
"""Neural Style Transfer Module"""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer:

    Public class attributes:
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'
    Class constructor: def __init__(self, style_image, content_image,
    alpha=1e4, beta=1):
    style_image - the image used as a style reference, stored as a
    numpy.ndarray
    content_image - the image used as a content reference, stored as a
    numpy.ndarray
    alpha - the weight for content cost
    beta - the weight for style cost
    if style_image is not a np.ndarray with the shape (h, w, 3), raise a
    TypeError with the message style_image must be a numpy.ndarray with
    shape (h, w, 3)
    if content_image is not a np.ndarray with the shape (h, w, 3), raise
    a TypeError with the message content_image must be a numpy.ndarray with
    shape (h, w, 3)
    if alpha is not a non-negative number, raise a TypeError with the message
    alpha must be a non-negative number
    if beta is not a non-negative number, raise a TypeError with the message
    beta must be a non-negative number
    Sets the instance attributes:
    style_image - the preprocessed style image
    content_image - the preprocessed content image
    alpha - the weight for content cost
    beta - the weight for style cost
    model - the Keras model used to calculate cost
    gram_style_features - a list of gram matrices calculated from the style
    layer outputs of the style image
    content_feature - the content layer output of the content image
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        if not isinstance(style_image, np.ndarray
                          ) or len(style_image.shape
                                   ) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray
                          ) or len(content_image.shape
                                   ) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        if not isinstance(var, (int, float)) or beta < 0:
            raise TypeError("var must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var

        self.load_model()

        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        image - a numpy.ndarray of shape (h, w, 3) containing the image to be
        scaled
        if image is not a np.ndarray with the shape (h, w, 3), raise a
        TypeError with the message image must be a numpy.ndarray with
        shape (h, w, 3)
        The scaled image should be a tf.tensor with the shape
        (1, h_new, w_new, 3) where max(h_new, w_new) == 512 and
        min(h_new, w_new) is scaled proportionately
        The image should be resized using bicubic interpolation
        After resizing, the image’s pixel values should be rescaled from the
        range [0, 255] to [0, 1].
        Returns: the scaled
        """
        if not isinstance(image, np.ndarray
                          ) or len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = w * h_new // h
        else:
            w_new = 512
            h_new = h * w_new // w

        scaled_image = tf.image.resize(image, tf.constant([h_new, w_new],
                                                          dtype=tf.int32),
                                       tf.image.ResizeMethod.BICUBIC)
        scaled_image = tf.reshape(scaled_image, (1, h_new, w_new, 3))
        scaled_image = tf.clip_by_value(scaled_image / 255, 0.0, 1.0)

        return scaled_image

    def load_model(self):
        """Creates the model used to calculate cost
        the model should use the VGG19 Keras model as a base
        the model’s input should be the same as the VGG19 input
        the model’s output should be a list containing the outputs of the
        VGG19 layers listed in style_layers followed by content _layer
        saves the model in the instance attribute model"""
        base_model = tf.keras.applications.VGG19(include_top=False)
        base_model.trainable = False

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        base_model.save('base_model')
        base_model = tf.keras.models.load_model('base_model',
                                                custom_objects=custom_objects)

        outputs = [base_model.get_layer(layer
                                        ).output for layer in self.style_layers]
        outputs.append(base_model.get_layer(self.content_layer).output)
        model = tf.keras.Model(base_model.inputs, outputs)

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """input_layer - an instance of tf.Tensor or tf.Variable of shape
        (1, h, w, c)containing the layer output whose gram matrix should be
        calculated
        if input_layer is not an instance of tf.Tensor or tf.Variable of
        rank 4, raise a TypeError with the message input_layer must be a tensor
        of rank 4
        Returns: a tf.Tensor of shape (1, c, c) containing the gram matrix of
        input_layer"""
        if not isinstance(input_layer, (tf.Variable, tf.Tensor)) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return result / num_locations

    def generate_features(self):
        """Extracts the features used to calculate neural style cost
        Sets the public instance attributes:
        gram_style_features - a list of gram matrices calculated from the style
        layer outputs of the style image
        content_feature - the content layer output of the content image
        """
        style_inputs = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_inputs = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        style_outputs = self.model(style_inputs)
        content_outputs = self.model(content_inputs)

        self.gram_style_features = [self.gram_matrix(
            style_layer) for style_layer in style_outputs[:-1]]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer
        style_output - tf.Tensor of shape (1, h, w, c) containing the layer
        style output of the generated image
        gram_target - tf.Tensor of shape (1, c, c) the gram matrix of the
        target style output for that layer
        if style_output is not an instance of tf.Tensor or tf.Variable of
        rank 4, raise a TypeError with the message style_output must be a
        tensor of rank 4
        if gram_target is not an instance of tf.Tensor or tf.Variable with
        shape (1, c, c), raise a TypeError with the message gram_target must
        be a tensor of shape [1, {c}, {c}] where {c} is the number of channels
        in style_output
        Returns: the layer’s style cost
        """
        if not isinstance(style_output, (tf.Variable, tf.Tensor)) or len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        _, _, _, c = style_output.shape

        if not isinstance(gram_target, (tf.Variable, tf.Tensor)) or gram_target.shape != (1, c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]")

        gram_style = self.gram_matrix(style_output)

        style_loss_layer = tf.math.reduce_mean(
            tf.square(
                tf.subtract(gram_target, gram_style)))

        return style_loss_layer

    def style_cost(self, style_outputs):
        """Calculates the style cost for generated image
        style_outputs - a list of tf.Tensor style outputs for the generated
        image
        if style_outputs is not a list with the same length as
        self.style_layers, raise a TypeError with the message style_outputs
        must be a list with a length of {l} where {l} is the length of
        self.style_layers
        each layer should be weighted evenly with all weights summing to 1
        Returns: the style cost
        """
        style_len = len(self.style_layers)
        if not isinstance(style_outputs, list
                          ) or len(style_outputs) != style_len:
            raise TypeError(
                f"style_outputs must be a list with a length of {style_len}")

        style_outputs_cost = tf.add_n([self.layer_style_cost(
            style_output, self.gram_style_features[i]
        ) for i, style_output in enumerate(style_outputs)])

        return style_outputs_cost * (self.beta / style_len)

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image
        content_output - a tf.Tensor containing the content output for the
        generated image
        if content_output is not an instance of tf.Tensor or tf.Variable with
        the same shape as self.content_feature, raise a TypeError with the
        message content_output must be a tensor of shape {s} where {s} is the
        shape of self.content_feature
        Returns: the content cost
        """
        content_shape = self.content_feature.shape

        if not isinstance(content_output, (tf.Variable, tf.Tensor)) or content_output.shape != content_shape:
            raise TypeError(
                f"content_output must be a tensor of shape {content_shape}")

        content_cost = tf.math.reduce_mean(
            tf.square(
                tf.subtract(self.content_feature, content_output)))

        return content_cost

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image
        generated_image - a tf.Tensor of shape (1, nh, nw, 3) containing the
        generated image
        if generated_image is not an instance of tf.Tensor or tf.Variable with
        the same shape as self.content_image, raise a TypeError with the
        message generated_image must be a tensor of shape {s} where {s} is the
        shape of self.content_image
        Returns: (J, J_content, J_style, J_var)
        J is the total cost
        J_content is the content cost
        J_style is the style cost
        J_var is the variational cost
        """
        content_shape = self.content_image.shape
        if not isinstance(generated_image, (tf.Variable, tf.Tensor)) or generated_image.shape != content_shape:
            raise TypeError(
                f"generated_image must be a tensor of shape {content_shape}")

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        outputs = self.model(preprocessed)
        content_output = outputs[-1]
        style_outputs = outputs[:-1]

        content_cost = self.content_cost(content_output)
        style_cost = self.style_cost(style_outputs)
        variation_cost = self.variational_cost(generated_image)
        total_cost = self.alpha * content_cost + \
            self.beta * style_cost + self.var * variation_cost

        return (total_cost, content_cost, style_cost, variation_cost)

    def compute_grads(self, generated_image):
        """Calculates the gradients for the tf.Tensor generated image of
        shape (1, nh, nw, 3)
        if generated_image is not an instance of tf.Tensor or tf.Variable with
        the same shape as self.content_image, raise a TypeError with the
        message generated_image must be a tensor of shape {s} where {s} is the
        shape of self.content_image
        Returns: gradients, J_total, J_content, J_style
        gradients is a tf.Tensor containing the gradients for the generated
        image
        J_total is the total cost for the generated image
        J_content is the content cost for the generated image
        J_style is the style cost for the generated image
        J_var is the variational cost for the generated image
        """
        content_shape = self.content_image.shape
        if not isinstance(generated_image, (tf.Variable, tf.Tensor)) or generated_image.shape != content_shape:
            raise TypeError(
                f"generated_image must be a tensor of shape {content_shape}")

        with tf.GradientTape() as tape:
            J_total, J_content, J_style, J_var = self.total_cost(
                generated_image)
        grad = tape.gradient(J_total, generated_image)

        return (grad, J_total, J_content, J_style, J_var)

    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9,
                       beta2=0.99):
        """iterations - the number of iterations to perform gradient descent
        over
        step - if not None, the step at which you should print information
        about the training, including the final iteration:
        print Cost at iteration {i}: {J_total}, content {J_content},
        style {J_style}
        i is the iteration
        J_total is the total cost
        J_content is the content cost
        J_style is the style cost
        lr - the learning rate for gradient descent
        beta1 - the beta1 parameter for gradient descent
        beta2 - the beta2 parameter for gradient descent
        if iterations is not an integer, raise a TypeError with the message
        iterations must be an integer
        if iterations is not positive, raise a ValueError with the message
        iterations must be positive
        if step is not None and not an integer, raise a TypeError with the
        message step must be an integer
        if step is not None and not positive or less than iterations , raise a
        ValueError with the message step must be positive and less than
        iterations
        if lr is not a float or an integer, raise a TypeError with the message
        lr must be a number
        if lr is not positive, raise a ValueError with the message lr must be
        positive
        if beta1 is not a float, raise a TypeError with the message beta1 must
        be a float
        if beta1 is not in the range [0, 1], raise a ValueError with the
        message beta1 must be in the range [0, 1]
        if beta2 is not a float, raise a TypeError with the message beta2 must
        be a float
        if beta2 is not in the range [0, 1], raise a ValueError with the
        message beta2 must be in the range [0, 1]
        gradient descent should be performed using Adam optimization
        the generated image should be initialized as the content image
        keep track of the best cost and the image associated with that cost
        Returns: generated_image, cost
        generated_image is the best generated image
        cost is the best cost
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")

        if not (isinstance(lr, float) or isinstance(lr, int)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not 0 <= beta1 <= 1:
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not 0 <= beta2 <= 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        opt = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2)
        generated_image = tf.Variable(self.content_image)
        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grad, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image)
            if step is not None and (i % step == 0 or i == iterations):
                print(
                    f"Cost at iteration {i}: {J_total}, content {J_content}, style {J_style}, var {J_var}")

            if J_total < best_cost:
                best_cost = J_total.numpy()
                best_image = generated_image.numpy()

            opt.apply_gradients([(grad, generated_image)])
            clip_image = tf.clip_by_value(generated_image, 0.0, 1.0)
            generated_image.assign(clip_image)

        return best_image, best_cost

    @staticmethod
    def variational_cost(generated_image):
        """Calculates the variational cost for the generated image
        generated_image - a tf.Tensor of shape (1, nh, nw, 3) containing the
        generated image
        Returns: the variational cost
        """
        # print(type(generated_image))
        # if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or generated_image.shape != 4 or generated_image.shape != 3:
        #     raise TypeError("image must be a tensor of rank 3 or 4")
        return tf.image.total_variation(generated_image)[0]
