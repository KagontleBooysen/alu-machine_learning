#!/usr/bin/env python3
"""Neural Style Transfer Module"""
import numpy as np
import tensorflow as tf


class NST:
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
        if not isinstance(alpha, float) and not isinstance(alpha, int
                                                           ) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, float) and not isinstance(beta, int
                                                          ) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        if not isinstance(var, float) and not isinstance(var, int
                                                          ) or var < 0:
            raise TypeError('var must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()
    @staticmethod
    def scale_image(image):
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
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
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
        model = tf.keras.models.Model(vgg.input, model_outputs)
        self.model = model
    @staticmethod
    def gram_matrix(input_layer):
        if not (isinstance(input_layer, tf.Tensor) or isinstance(input_layer, tf.Variable)) or input_layer.shape.ndims != 4:
            raise TypeError('input_layer must be a tensor of rank 4')
        _, nh, nw, _ = input_layer.shape.dims
        G = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        return G / tf.cast(nh * nw, tf.float32)

    def generate_features(self):
        preprocessed_s = tf.keras.applications.vgg19.preprocess_input(self.style_image * 255)
        preprocessed_c = tf.keras.applications.vgg19.preprocess_input(self.content_image * 255)
        style_features = self.model(preprocessed_s)[:-1]
        self.content_feature = self.model(preprocessed_c)[-1]
        self.gram_style_features = [self.gram_matrix(style_feature) for style_feature in style_features]
        
    def layer_style_cost(self, style_output, gram_target):
        if not (isinstance(style_output, tf.Tensor) or isinstance(style_output, tf.Variable)) or style_output.shape.ndims != 4:
            raise TypeError('style_output must be a tensor of rank 4')
        m, _, _, nc = style_output.shape.dims
        if not (isinstance(gram_target, tf.Tensor) or isinstance(gram_target, tf.Variable)) or gram_target.shape.dims != [m, nc, nc]:
            raise TypeError('gram_target must be a tensor of shape [{}, {}, {}]'.format(m, nc, nc))
        gram_style = self.gram_matrix(style_output)
        return tf.reduce_sum(tf.square(gram_style - gram_target)) / tf.square(tf.cast(nc, tf.float32))
    def style_cost(self, style_outputs):
        if type(style_outputs) is not list or len(style_outputs) != len(self.style_layers):
            raise TypeError('style_outputs must be a list with a length of {}'.format(len(self.style_layers)))
        J_style = tf.add_n([self.layer_style_cost(style_outputs[i], self.gram_style_features[i]) for i in range(len(style_outputs))])
        J_style /= tf.cast(len(style_outputs), tf.float32)
        return J_style
    def content_cost(self, content_output):
        if not (isinstance(content_output, tf.Tensor) or isinstance(content_output, tf.Variable)) or content_output.shape.dims != self.content_feature.shape.dims:
            raise TypeError('content_output must be a tensor of shape {}'.format(self.content_feature.shape))
        _, nh, nw, nc = content_output.shape.dims
        return tf.reduce_sum(tf.square(content_output - self.content_feature)) / tf.cast(nh * nw * nc, tf.float32)
    @staticmethod
    def variational_cost(generated_image):
        if not (isinstance(generated_image, tf.Tensor) or isinstance(generated_image, tf.Variable)) or not 3 <= generated_image.shape.ndims <= 4:
            raise TypeError('image must be a tensor of rank 3 or 4')
        return tf.reduce_sum(tf.image.total_variation(generated_image))
    def total_cost(self, generated_image):
        if not (isinstance(generated_image, tf.Tensor) or isinstance(generated_image, tf.Variable)) or generated_image.shape.dims != self.content_image.shape.dims:
            raise TypeError('generated_image must be a tensor of shape {}'.format(self.content_image.shape))
        preprocessed = tf.keras.applications.vgg19.preprocess_input(generated_image * 255)
        model_outputs = self.model(preprocessed)
        style_outputs = [style_layer for style_layer in model_outputs[:-1]]
        content_output = model_outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J_var = self.variational_cost(generated_image)
        J = (self.alpha * J_content) + (self.beta * J_style) + (self.var * J_var)
        return J, J_content, J_style, J_var
    def compute_grads(self, generated_image):
        if not (isinstance(generated_image, tf.Tensor) or isinstance(generated_image, tf.Variable)) or generated_image.shape.dims != self.content_image.shape.dims:
            raise TypeError('generated_image must be a tensor of shape {}'.format(self.content_image.shape))
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J, J_content, J_style, J_var = self.total_cost(generated_image)
        grads = tape.gradient(J, generated_image)
        return grads, J, J_content, J_style, J_var
    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99):
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be positive')
        if step is not None and not isinstance(step, int):
            raise TypeError('step must be an integer')
        if step is not None and not 0 < step < iterations:
            raise ValueError('step must be positive and less than iterations')
        if not isinstance(lr, (float, int)):
            raise TypeError('lr must be a number')
        if lr <= 0:
            raise ValueError('lr must be positive')
        if not isinstance(beta1, float):
            raise TypeError('beta1 must be a float')
        if not 0 <= beta1 <= 1:
            raise ValueError('beta1 must be in the range [0, 1]')
        if not isinstance(beta2, float):
            raise TypeError('beta2 must be a float')
        if not 0 <= beta2 <= 1:
            raise ValueError('beta2 must be in the range [0, 1]')

        content_image_tensor = tf.convert_to_tensor(self.content_image, dtype=tf.float32)
        generated_image = tf.Variable(self.content_image)
        optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
        best_cost = np.inf
        best_image = None

        for i in range(iterations):
            grads, J, J_content, J_style, J_var = self.compute_grads(generated_image)
            print("for",J < best_cost)
            if step is not None and i % step == 0:
                print("Cost at iteration {}: {}, content {}, style {}, var {}".format(i, J, J_content, J_style, J_var))
            if J < best_cost:
                best_cost = J.numpy()
                best_image = generated_image.numpy()[0]
            optimizer.apply_gradients([(grads, generated_image)])
            generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))

        _, J, J_content, J_style, J_var = self.compute_grads(generated_image)
        print("last",J < best_cost)
        if J < best_cost:
            best_cost = J.numpy()
            best_image = generated_image.numpy()[0]
        if step is not None:
            print("Cost at iteration {}: {}, content {}, style {}, var {}".format(iterations, J, J_content, J_style, J_var))
        return best_image, best_cost
