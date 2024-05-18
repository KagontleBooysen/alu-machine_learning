#!/usr/bin/env python3
"""nst_class"""
import numpy as np
import tensorflow as tf


class NST:
    """class NST"""
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """initialization"""
        tf.executing_eagerly()
        if type(style_image) is not np.ndarray or style_image.ndim != 3\
                or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if type(content_image) is not np.ndarray or content_image.ndim != 3\
                or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
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
        """rescales an image such that its pixels values are
          between 0 and 1 and its largest side is 512 pixels
          """
        if type(image) is not np.ndarray or\
                image.shape[-1] != 3:
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')
        h, w, c = image.shape
        maxd = max(h, w)
        new_h = int(h * (512/maxd))
        new_w = int(w * (512/maxd))
        image = tf.expand_dims(image, axis=0)
        resized = tf.image.resize(image, size=(new_h, new_w), method="bicubic")
        resized /= 255
        return tf.clip_by_value(resized, 0.0, 1.0)

    def load_model(self):
        """
        Load VGG19 model
        :return: The model
        """
        vgg19 = tf.keras.applications.VGG19(include_top=False)
        for layer in vgg19.layers:
            layer.trainable = False
        vgg19.save("model.h5")
        model = tf.keras.models.load_model(
            "model.h5",
            # change MaxPooling2D to AveragePooling2D like in the paper
            custom_objects={
                "MaxPooling2D": tf.keras.layers.AveragePooling2D()
            })

        outputs = ([model.get_layer(layer).output
                   for layer in self.style_layers]
                   + [model.get_layer(self.content_layer).output])
        self.model = tf.keras.models.Model(model.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """ gram matrix """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or\
                tf.rank(input_layer).numpy() != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        ndata, h, w, c = tf.shape(input_layer).numpy()
        F = tf.reshape(input_layer, (ndata, h * w, c))
        gram = tf.matmul(F, F, transpose_a=True)
        gram /= (h * w)
        return gram

    def generate_features(self):
        """ extract features used to calculate neural style cost """
        vgg19 = tf.keras.applications.vgg19
        content = vgg19.preprocess_input(self.content_image * 255)
        style = vgg19.preprocess_input(self.style_image * 255)
        out_c = self.model(content)
        outputs = self.model(style)
        sty = outputs[:len(outputs) - 1]
        cont = out_c[len(out_c) - 1]
        gram_style_features = []
        for ft in sty:
            gram_style_features.append(self.gram_matrix(ft))
        self.gram_style_features = gram_style_features
        self.content_feature = cont

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        Returns: the layer’s style cost
        """
        style_output = self.gram_matrix(style_output)
        style_loss = tf.reduce_mean((style_output-gram_target)**2)
        return style_loss

    def style_cost(self, style_outputs):
        """
        Calculates the style cost
        Returns: the style cost
        """
        if len(style_outputs) != len(self.style_layers):
            raise TypeError("style_outputs must be a list with a length of \
            {}").format(len(style_layers))
        style_cost = 0
        weight_per_layer = 1.0 / len(style_outputs)
        for i in range(len(style_outputs)):
            layer_style_cost = self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i])
            style_cost += tf.reduce_sum(layer_style_cost) * weight_per_layer
        return style_cost

    def content_cost(self, content_output):
        """
        Calculate the content cost
        Returns: the content cost
        """
        content_cost = tf.reduce_mean((content_output -
                                       self.content_feature)**2)
        return content_cost

    def total_cost(self, generated_image):
        """
        Calculates the total cost for the generated image
        Returns: total cost, content cost, style cost
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != self.content_image.shape:
            raise TypeError("generated_image must be a tensor \
            of shape {}".format(self.content_image.shape))
        preprocecced = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        model_outputs = self.model(preprocecced)
        content_cost = self.content_cost(model_outputs[-1])
        style_cost = self.style_cost(model_outputs[:5])
        total_cost = content_cost*self.alpha + style_cost*self.beta
        return total_cost, content_cost, style_cost

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the tf.Tensor generated image
        Returns: gradients, J_total, J_content, J_style
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != self.content_image.shape:
            raise TypeError("generated_image must be a tensor \
        of shape {}".format(self.content_image.shape))
        J_total, J_content, J_style = self.total_cost(generated_image)
        with tf.GradientTape() as tape:
            loss, _, _ = self.total_cost(generated_image)
        grads = tape.gradient(loss, generated_image)
        return grads, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99):
        """generates image from noise"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be positive")
        if step is not None and type(step) is not int:
            raise TypeError("step must be an integer")
        if step is not None and (step < 0 or step > iterations):
            raise ValueError("step must be positive and less than iterations")
        if type(lr) is not float:
            raise TypeError("lr must be a number")
        if lr < 0:
            raise ValueError("lr must be positive")
        if type(beta1) is not float:
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if type(beta2) is not float:
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2)
        generated_image = tf.Variable(self.content_image)
        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grad, J_total, J_content, J_style = self.compute_grads(
                generated_image)
            if step is not None and (i % step == 0 or i == iterations):
                print(
                    f"Cost at iteration {i}: {J_total}, content {J_content}, style {J_style}")

            if J_total < best_cost:
                best_cost = J_total.numpy()
                best_image = generated_image.numpy()

            if i != iterations:
                opt.apply_gradients([(grad, generated_image)])
                clip_image = tf.clip_by_value(generated_image, 0.0, 1.0)
                generated_image.assign(clip_image)
        #change best image shape to (h, w, 3)
        best_image = best_image.reshape(self.content_image.shape)
        import pickle as pkl
        pkl.dump(best_image, open('best_image.pkl', 'wb'))
        return best_image, best_cost
