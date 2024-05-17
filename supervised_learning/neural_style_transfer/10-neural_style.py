#!/usr/bin/env python3
"""Module class NST."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """
        Class constructor.

        Args:
            style_image (ndarray): image used as a style reference
            content_image (ndarray): image used as a content reference
            alpha (float, optional): weight for content cost. Defaults to 1e4.
            beta (int, optional): weight for style cost. Defaults to 1.
            var (int, optional): weight for the variational cost. Defaults to
                                 10.
        """
        if not isinstance(style_image, np.ndarray):
            raise TypeError("style_image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        if style_image.ndim != 3 or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        if not isinstance(content_image, np.ndarray):
            raise TypeError("content_image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        if content_image.ndim != 3 or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        if not isinstance(var, (int, float)) or var < 0:
            raise TypeError("var must be a non-negative number")
        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1 and\
        its largest side is 512 pixels.

        Args:
            image (ndarray): contains the image to be scaled
        Returns:
            scaled image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        if image.ndim != 3 or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape\
                            (h, w, 3)")
        max_dim = 512
        long_dim = max(image.shape[:-1])
        scale = max_dim / long_dim
        new_shape = tuple(map(lambda x: int(scale * x), image.shape[:-1]))
        image = image[tf.newaxis, :]
        image = tf.image.resize_bicubic(image, new_shape)
        image = image / 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        return image

    def load_model(self):
        """Create the model used to calculate cost."""
        base_vgg = tf.keras.applications.VGG19(include_top=False,
                                               weights='imagenet',
                                               input_tensor=None,
                                               input_shape=None, pooling=None,
                                               classes=1000)
        custom_object = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        base_vgg.save('base_vgg')
        vgg = tf.keras.models.load_model('base_vgg',
                                         custom_objects=custom_object)
        for layer in vgg.layers:
            layer.trainable = False
        style_outputs = [vgg.get_layer(name).output for name in
                         self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate gram matrices.

        Args:
            input_layer (tf.Tensor or tf.Variable): contains the layer output
                                                    whose gram matrix should
                                                    be calculated
        Returns:
            tf.Tensor containing the gram matrix of input_layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        result = result / num_locations
        return result

    def generate_features(self):
        """Extract the features used to calculate neural style cost."""
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        outputs_style = self.model(style_image)
        style_outputs = outputs_style[:-1]
        outputs_content = self.model(content_image)
        content_output = outputs_content[-1]
        self.gram_style_features = [self.gram_matrix(style_output) for
                                    style_output in style_outputs]
        self.content_feature = content_output

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculate the style cost for a single layer.

        Args:
            style_output (Tensor): contains the layer style output of the
                                   generated image
            gram_target (Tensor): gram matrix of the target style output for
                                  that layer
        Returns:
            layer's style cost
        """
        c = style_output.shape[-1]
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError("style_output must be a tensor of rank 4")
        if len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError("gram_target must be a tensor of shape\
                [1, {}, {}]".format(c, c))
        if gram_target.shape != (1, c, c):
            raise TypeError("gram_target must be a tensor of shape\
                [1, {}, {}]".format(c, c))
        gram_style = self.gram_matrix(style_output)
        style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))
        return style_cost

    def style_cost(self, style_outputs):
        """
        Calculate the style cost.

        Args:
            style_outputs (Tensor): style outputs for the generated image
        Returns:
            style cost
        """
        if not isinstance(style_outputs, list):
            raise TypeError("style_outputs must be a list with a length of\
                {}".format(len(self.style_layers)))
        if len(self.style_layers) != len(style_outputs):
            raise TypeError("style_outputs must be a list with a length of\
                {}".format(len(self.style_layers)))
        style_costs = []
        weight = 1 / len(self.style_layers)
        for style_output, gram_target in zip(style_outputs,
                                             self.gram_style_features):
            layer_style_cost = self.layer_style_cost(style_output, gram_target)
            weighted_layer_style_cost = weight * layer_style_cost
            style_costs.append(weighted_layer_style_cost)
        style_cost = tf.add_n(style_costs)
        return style_cost

    def content_cost(self, content_output):
        """
        Calculate the content cost.

        Args:
            content_output (Tensor): contains the content output for the
                                     generated image
        Returns:
            content cost
        """
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError("content_output must be a tensor of shape\
                {}".format(self.content_feature.shape))
        if content_output.shape != self.content_feature.shape:
            raise TypeError("content_output must be a tensor of shape\
                {}".format(self.content_feature.shape))
        content_cost = tf.reduce_mean(tf.square(content_output -
                                                self.content_feature))
        return content_cost

    def total_cost(self, generated_image):
        """
        Calculate the total cost.

        Args:
            generated_image (Tensor): contains the generated image
        Returns:
            (J, J_content, J_style, J_var)
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError("generated_image must be a tensor of shape\
                {}".format(self.content_image.shape))
        if generated_image.shape != self.content_image.shape:
            raise TypeError("generated_image must be a tensor of shape\
                {}".format(self.content_image.shape))
        generated_image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        outputs_generated = self.model(generated_image)
        style_outputs = outputs_generated[:-1]
        content_output = outputs_generated[-1]
        style_cost = self.style_cost(style_outputs)
        content_cost = self.content_cost(content_output)
        var_cost = self.variational_cost(generated_image)
        total_cost = (self.alpha * content_cost + self.beta * style_cost +
                      self.var * var_cost)
        return (total_cost, content_cost, style_cost, var_cost)

    def compute_grads(self, generated_image):
        """
        Compute the gradients for the generated image.

        Args:
            generated_image (Tensor): contains the generated image
        Returns:
            gradients, J_total, J_content, J_style
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError("generated_image must be a tensor of shape\
                {}".format(self.content_image.shape))
        if generated_image.shape != self.content_image.shape:
            raise TypeError("generated_image must be a tensor of shape\
                {}".format(self.content_image.shape))
        with tf.GradientTape() as tape:
            loss = self.total_cost(generated_image)
            total_cost, content_cost, style_cost = loss
        gradients = tape.gradient(total_cost, generated_image)
        return (gradients, total_cost, content_cost, style_cost)

    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9,
                       beta2=0.99):
        """
        Generate the neural style transfered image.

        Args:
            iterations (int, optional): number of iterations to perform
                                        gradient descent over. Defaults to
                                        1000.
            step (int, optional): step at which you should print information
                                  about the training. Defaults to None.
            lr (float, optional): learning rate for gradient descent. Defaults
                                  to 0.01.
            beta1 (float, optional): beta1 parameter for gradient descent.
                                     Defaults to 0.9.
            beta2 (float, optional): beta2 parameter for gradient descent.
                                     Defaults to 0.99.
        Returns:
            generated_image, cost
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and less than\
                    iterations")
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")
        generated_image = tf.Variable(self.content_image)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1,
                                           beta2=beta2)
        prev_total_cost = float('inf')
        prev_image = generated_image
        for i in range(iterations + 1):
            computed = self.compute_grads(generated_image)
            (gradients, total_cost, content_cost, style_cost,
             var_cost) = computed
            if i % step == 0 or i == iterations:
                print("Cost at iteration {}: {}, content {}, style {}, var\
                    {}".format(i, total_cost, content_cost, style_cost,
                               var_cost))
            if i != iterations:
                optimizer.apply_gradients([(gradients, generated_image)])
                clipped_image = tf.clip_by_value(generated_image,
                                                 clip_value_min=0,
                                                 clip_value_max=1)
                generated_image.assign(clipped_image)
            if total_cost <= prev_total_cost:
                prev_total_cost = total_cost
                prev_image = generated_image
        cost = prev_total_cost.numpy()
        generated_image = prev_image[0].numpy()
        return (generated_image, cost)

    @staticmethod
    def variational_cost(generated_image):
        """
        Calculate the variational cost for the generated image.

        Args:
            generated_image (Tensor): contains the generated image
        Returns:
            variational cost
        """
        loss = tf.image.total_variation(generated_image)[0]
        return loss
