# models/teacher_model.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class StyleTransferModel:
    def __init__(self,
                 content_weight=1e4,
                 style_weight=1e3,
                 tv_weight=5,
                 num_iterations=2000,
                 target_size=(512, 512)):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.num_iterations = num_iterations
        self.target_size = target_size

        self.model, self.content_layers, self.style_layers = self._get_model()
        self.num_content_layers = len(self.content_layers)

    def _get_model(self):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        content_layers = ['block5_conv2']
        style_layers = ['block3_conv1', 'block4_conv1', 'block5_conv1']
        outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
        model = tf.keras.Model(vgg.input, outputs)
        return model, content_layers, style_layers

    def _gram_matrix(self, tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
        shape = tf.shape(tensor)
        num_locations = tf.cast(shape[1] * shape[2], tf.float32)
        return result / num_locations

    def _compute_loss(self, generated_image, content_features, style_features):
        outputs = self.model(generated_image)
        content_output = outputs[:self.num_content_layers]
        style_output = outputs[self.num_content_layers:]

        content_loss = tf.add_n([
            tf.reduce_mean((gen - real) ** 2)
            for gen, real in zip(content_output, content_features)
        ])

        style_loss = tf.add_n([
            tf.reduce_mean((self._gram_matrix(gen) - self._gram_matrix(real)) ** 2)
            for gen, real in zip(style_output, style_features)
        ])
        style_loss /= len(style_output)

        total_variation_loss = tf.image.total_variation(generated_image)

        total_loss = (self.content_weight * content_loss +
                      self.style_weight * style_loss +
                      self.tv_weight * total_variation_loss)
        return total_loss

    def _get_feature_representations(self, content_image, style_image):
        content_outputs = self.model(content_image)
        style_outputs = self.model(style_image)
        content_features = content_outputs[:self.num_content_layers]
        style_features = style_outputs[self.num_content_layers:]
        return content_features, style_features

    @tf.function
    def _train_step(self, optimizer, content_features, style_features, generated_image):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(generated_image, content_features, style_features)
        grad = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, -128.0, 128.0))

    def stylize(self, content_image, style_image, visualize=False):
        content_features, style_features = self._get_feature_representations(content_image, style_image)
        generated_image = tf.Variable(content_image, dtype=tf.float32)

        # 💡 Create the optimizer ONCE with generated_image to initialize its state
        optimizer = tf.optimizers.Adam(learning_rate=5.0)
        _ = optimizer.apply_gradients([(tf.zeros_like(generated_image), generated_image)])  # dummy init

        best_img = None
        best_loss = float('inf')

        for i in range(self.num_iterations + 1):
            self._train_step(optimizer, content_features, style_features, generated_image)

            if loss := self._compute_loss(generated_image, content_features, style_features):
                if loss < best_loss:
                    best_loss = loss
                    best_img = generated_image.numpy()
        return self.deprocess_image(best_img)

    def load_and_process_image(self, path):
        img = load_img(path, target_size=self.target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return preprocess_input(img)

    def deprocess_image(self, processed_img):
        x = processed_img.copy()
        x = x.reshape((x.shape[1], x.shape[2], 3))
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]
        return np.clip(x, 0, 255).astype('uint8')
