#!/usr/bin/env python3
""" NTS """
import numpy as np
import tensorflow as tf


class NST:
    """ class NTS """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        - style_image - the image used as a style reference,
            stored as a numpy.ndarray
        - content_image - the image used as a content reference,
            stored as a numpy.ndarray
        - alpha - the weight for content cost
        - beta - the weight for style cost
        if style_image is not a np.ndarray with the shape (h, w, 3),
            raise a TypeError with the message:
            style_image must be a numpy.ndarray with shape (h, w, 3)
        if content_image is not a np.ndarray with the shape (h, w, 3),
            raise a TypeError with the message:
            content_image must be a numpy.ndarray with shape (h, w, 3)
        if alpha is not a non-negative number,
            raise a TypeError with the message:
            alpha must be a non-negative number
        if beta is not a non-negative number,
            raise a TypeError with the message:
            beta must be a non-negative number
        Sets Tensorflow to execute eagerly
        Sets the instance attributes:
            style_image - the preprocessed style image
            content_image - the preprocessed content image
            alpha - the weight for content cost
            beta - the weight for style cost
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or style_image.shape[2] != 3):
            msg = "style_image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(msg)

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or content_image.shape[2] != 3):
            msg = "content_image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(msg)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        tf.enable_eager_execution()
        self.content_image = self.scale_image(content_image)
        self.style_image = self.scale_image(style_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """ rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels """

        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or image.shape[2] != 3):
            msg = "image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(msg)

        new_h = 512
        new_w = 512
        if image.shape[0] > image.shape[1]:
            new_w = int(image.shape[1] * 512 / image.shape[0])

        elif image.shape[0] < image.shape[1]:
            new_h = int(image.shape[0] * 512 / image.shape[1])

        mth = tf.image.ResizeMethod.BICUBIC
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bicubic(image, (new_h, new_w),
                                        align_corners=False)

        image = image / 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

        return image

    def load_model(self):
        """ loads the model for neural style transfer """
        vgg_pre = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                    weights='imagenet')

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg_pre.save("base_model")
        vgg = tf.keras.models.load_model("base_model",
                                         custom_objects=custom_objects)
        for layer in vgg.layers:
            layer.trainable = False

        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(vgg.input, model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """ calculates gram matrices """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")
        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        gram = tf.expand_dims(gram, 0)
        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """ extracts the features used to calculate neural style cost """
        vgg19 = tf.keras.applications.vgg19
        content = vgg19.preprocess_input(self.content_image * 255)
        style = vgg19.preprocess_input(self.style_image * 255)
        out_content = self.model(content)
        outputs = self.model(style)
        list_gram = []
        for out in outputs[:-1]:
            list_gram = list_gram + [self.gram_matrix(out)]

        self.gram_style_features = list_gram

        self.content_feature = out_content[-1]

    def layer_style_cost(self, style_output, gram_target):
        """ calculate the style cost for a single layer """

        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")
        c = int(style_output.shape[-1])
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            m = ("gram_target must be a tensor of shape [1, {}, {}]"
                 .format(c, c))
            raise TypeError(m)

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """  calculate the style cost """
        if (not type(style_outputs) is list
                or len(self.style_layers) != len(style_outputs)):
            le = len(self.style_layers)
            m = "style_outputs must be a list with a length of {}".format(le)
            raise TypeError(m)
        weight_per_style_layer = 1.0 / float(len(self.style_layers))
        loss = 0.0
        for target, style in zip(self.gram_style_features, style_outputs):
            loss = loss + (self.layer_style_cost(style, target)
                           * weight_per_style_layer)

        return loss

    def content_cost(self, content_output):
        """ Calculates the content cost for the generated image """
        s = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError("content_output must be a tensor of shape {}"
                            .format(s))
        if self.content_feature.shape != content_output.shape:
            raise TypeError("content_output must be a tensor of shape {}"
                            .format(s))

        if len(content_output.shape) == 3:
            content_output = tf.expand_dims(content_output, 0)

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """ Calculates the total cost for the generated image """
        s = self.content_image.shape
        m = "generated_image must be a tensor of shape {}".format(s)
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(m)
        if generated_image.shape != s:
            raise TypeError(m)

        vgg19 = tf.keras.applications.vgg19
        generated = vgg19.preprocess_input(generated_image * 255)
        out_generated = self.model(generated)
        style_cost = self.style_cost(out_generated[:-1])
        content_cost = self.content_cost(out_generated[-1])
        cost = self.alpha * content_cost + self.beta * style_cost

        return cost, content_cost, style_cost

    def compute_grads(self, generated_image):
        """ Calculates the gradients for the tf.Tensor
            generated image of shape (1, nh, nw, 3)
        """
        s = self.content_image.shape
        m = "generated_image must be a tensor of shape {}".format(s)
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(m)
        if generated_image.shape != s:
            raise TypeError(m)
        with tf.GradientTape() as tape:
            J_total, J_content, J_style = self.total_cost(generated_image)
        # Compute gradients wrt input image
        # total_loss = all_loss[0]
        grad = tape.gradient(J_total, generated_image)
        return grad, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """ generate the neural style transfered image """
        if not type(iterations) == int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not type(step) == int:
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                m = "step must be positive and less than iterations"
                raise ValueError(m)
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if type(beta1) != float:
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if type(beta2) != float:
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = self.content_image
        generated_image = tf.Variable(generated_image, dtype=tf.float32)
        opt = tf.train.AdamOptimizer(learning_rate=lr,
                                     beta1=beta1,
                                     beta2=beta2)
        best_loss, best_img = float('inf'), None
        for i in range(iterations):
            grad, J_total, J_content, J_style = (self.compute_grads
                                                 (generated_image))
            opt.apply_gradients([(grad, generated_image)])
            clipped = tf.clip_by_value(generated_image, 0, 1)
            generated_image.assign(clipped)
            if J_total < best_loss:
                best_loss = J_total.numpy()
                best_img = generated_image.numpy()
            if step is not None:
                if step == 0 or i % step == 0 or i == iterations - 1:
                    m = ("Cost at iteration {}: {}, content {}, style {}"
                         .format(i, J_total, J_content, J_style))
                    print(m)

        return best_img[-1, :, :], best_loss
