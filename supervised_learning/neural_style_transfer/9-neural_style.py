#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer
"""

import numpy as np
import tensorflow as tf

class NST:
    """
    Performs tasks for Neural Style Transfer

    public class attributes:
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'

    instance attributes:
        style_image: preprocessed style image
        content_image: preprocessed style image
        alpha: weight for content cost
        beta: weight for style cost
        model: the Keras model used to calculate cost
        gram_style_features: list of gram matrices from style layer outputs
        content_feature: the content later output of the content image

    class constructor:
        def __init__(self, style_image, content_image, alpha=1e4, beta=1)

    static methods:
        def scale_image(image):
            rescales an image so the pixel values are between 0 and 1
                and the largest side is 512 pixels
        def gram_matrix(input_layer):
            calculates gram matrices

    public instance methods:
        def load_model(self):
            creates model used to calculate cost from VGG19 Keras base model
        def generate_features(self):
            extracts the features used to calculate neural style cost
        def layer_style_cost(self, style_output, gram_target):
            calculates the style cost for a single layer
        def style_cost(self, style_outputs):
            calculates the style cost for generated image
        def content_cost(self, content_output):
            calculates the content cost for the generated image
        def total_cost(self, generated_image):
            calculates the total cost for the generated image
        def compute_grads(self, generated_image):
            calculates the gradients for the generated image
        def generate_image(self, iterations=1000, step=None, lr=0.01,
            beta1=0.9, beta2=0.99):
            generates the neural style transfered image
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer class

        parameters:
            style_image [numpy.ndarray with shape (h, w, 3)]:
                image used as style reference
            content_image [numpy.ndarray with shape (h, w, 3)]:
                image used as content reference
            alpha [float]: weight for content cost
            beta [float]: weight for style cost

        Raises TypeError if input are in incorrect format
        Sets TensorFlow to execute eagerly
        Sets instance attributes
        """
        if type(style_image) is not np.ndarray or \
           len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if type(content_image) is not np.ndarray or \
           len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape
        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (type(alpha) is not float and type(alpha) is not int) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.compat.v1.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
            and its largest side is 512 pixels

        parameters:
            image [numpy.ndarray of shape (h, w, 3)]:
                 image to be rescaled

        Scaled image should be tf.tensor with shape (1, h_new, w_new, 3)
            where max(h_new, w_new) is 512 and
            min(h_new, w_new) is scaled proportionately
        Image should be resized using bicubic interpolation.
        Image's pixels should be rescaled from range [0, 255] to [0, 1].

        returns:
            the scaled image
        """
        if type(image) is not np.ndarray or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, c = image.shape
        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize(np.expand_dims(image, axis=0),
                                  size=(h_new, w_new),
                                  method='bicubic')
        rescaled = resized / 255.0
        return tf.clip_by_value(rescaled, 0.0, 1.0)

    def load_model(self):
        """
        Creates the model used to calculate cost from VGG19 Keras base model

        Model's input should match VGG19 input
        Model's output should be a list containing outputs of VGG19 layers
            listed in style_layers followed by content_layer

        Saves the model in the instance attribute model
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        model_outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(vgg.input, model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates gram matrices

        parameters:
            input_layer [an instance of tf.Tensor or tf.Variable
                of shape (1, h, w, c)]:
                contains the layer output to calculate gram matrix for

        returns:
            tf.Tensor of shape (1, c, c) containing gram matrix of input_layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, (-1, c))
        gram = tf.matmul(features, features, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        gram /= tf.cast(h * w, tf.float32)
        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost

        Sets public instance attribute:
            gram_style_features and content_feature
        """
        vgg_preprocessor = tf.keras.applications.vgg19.preprocess_input

        style_image = vgg_preprocessor(self.style_image * 255)
        content_image = vgg_preprocessor(self.content_image * 255)

        style_outputs = self.model(style_image)[:-1]
        content_output = self.model(content_image)[-1]

        self.gram_style_features = [self.gram_matrix(output) for output in style_outputs]
        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer

        parameters:
            style_output [tf.Tensor of shape (1, h, w, c)]:
                contains the layer style output of the generated image
            gram_target [tf.Tensor of shape (1, c, c)]:
                the gram matrix of the target style output for that layer

        returns:
            the layer's style cost
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           len(gram_target.shape) != 3:
            raise TypeError(
                "gram_target must be a tensor of shape [1, c, c]")

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for generated image

        parameters:
            style_outputs [list of tf.Tensors]:
                contains stye outputs for the generated image

        returns:
            the style cost
        """
        if type(style_outputs) is not list or len(style_outputs) != len(self.style_layers):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(len(self.style_layers)))

        total_style_cost = 0
        weight_per_style_layer = 1.0 / float(len(self.style_layers))
        for style_output, gram_target in zip(style_outputs, self.gram_style_features):
            total_style_cost += weight_per_style_layer * self.layer_style_cost(style_output, gram_target)
        return total_style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for generated image

        parameters:
            content_output [tf.Tensor]:
                contains content output for the generated image

        returns:
            the style cost
        """
        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image

        parameters:
            generated_image [tf.Tensor of shape (1, nh, nw, 3)]:
                contains the generated image

        returns:
            (J, J_content, J_style) [tuple]:
                J: total cost
                J_content: content cost
                J_style: style cost
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           len(generated_image.shape) != 4 or generated_image.shape[0] != 1:
            raise TypeError(
                "generated_image must be a tensor of shape (1, nh, nw, 3)")

        vgg_preprocessor = tf.keras.applications.vgg19.preprocess_input
        preprocessed_image = vgg_preprocessor(generated_image * 255)
        outputs = self.model(preprocessed_image)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J_total = self.alpha * J_content + self.beta * J_style
        return J_total, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the generated image

        parameters:
            generated_image [tf.Tensor of shape (1, nh, nw, 3)]:
                contains the generated image

        returns:
            gradients, J_total, J_content, J_style
                gradients [tf.Tensor]: contains gradients for generated image
                J_total: total cost for the generated image
                J_content: content cost
                J_style: style cost
        """
        with tf.GradientTape() as tape:
            J_total, J_content, J_style = self.total_cost(generated_image)
        gradients = tape.gradient(J_total, generated_image)
        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """
        Generates the neural style transferred image

        parameters:
            iterations [int]:
                number of iterations to perform gradient descent over
            step [int or None]:
                step at which to print information about training
                prints:
                    i: iteration
                    J_total: total cost for generated image
                    J_content: content cost
                    J_style: style cost
            lr [float]:
                learning rate for gradient descent
            beta1 [float]:
                beta1 parameter for gradient descent
            beta2 [float[:
                beta2 parameter for gradient descent

        Gradient descent should be performed using Adam optimization.
        The generated image should be initialized as the content image.
        Keep track of the best cost and the image associated with that cost.

        returns:
            generated_image, cost
                generated_image: best generated image
                cost: best cost
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None and (type(step) is not int or step <= 0):
            raise TypeError("step must be a positive integer")
        if step is not None and step > iterations:
            raise ValueError("step must be less than or equal to iterations")
        if type(lr) is not float and type(lr) is not int:
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if type(beta1) is not float or not (0 <= beta1 <= 1):
            raise ValueError("beta1 must be a float in the range [0, 1]")
        if type(beta2) is not float or not (0 <= beta2 <= 1):
            raise ValueError("beta2 must be a float in the range [0, 1]")

        generated_image = tf.Variable(self.content_image, dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate=lr, beta1=beta1, beta2=beta2)
        best_cost = float('inf')
        best_image = None

        for i in range(iterations):
            grads, J_total, J_content, J_style = self.compute_grads(generated_image)
            optimizer.apply_gradients([(grads, generated_image)])
            clipped_image = tf.clip_by_value(generated_image, 0.0, 1.0)
            generated_image.assign(clipped_image)

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step and (i % step == 0 or i == iterations - 1):
                print("Iteration {}:".format(i))
                print("total cost = {}".format(J_total))
                print("content cost = {}".format(J_content))
                print("style cost = {}".format(J_style))

        return best_image, best_cost

