# style_transfer_logic.py
import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image

# --- Helper Functions ---

def load_and_process_image(path_to_img, target_size=(320, 320)):
    try:
        img = tf.keras.preprocessing.image.load_img(path_to_img, target_size=target_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return tf.keras.applications.vgg19.preprocess_input(img) # VGG19 specific preprocessing
    except Exception as e:
        print(f"Error loading image {path_to_img}: {e}")
        return None

def deprocess_image(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    if x.shape[-1] != 3:
        # Handle potential grayscale or alpha channels gracefully if needed,
        # For now, assume 3 channels or raise error
        if x.shape[-1] == 1: # Grayscale
             x = np.concatenate([x, x, x], axis=-1)
        elif x.shape[-1] == 4: # RGBA
             x = x[:, :, :3] # Drop alpha
        else:
            raise ValueError(f"Invalid input shape for deprocessing: {x.shape}")

    # Inverse of VGG19 preprocess_input
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR -> RGB
    return np.clip(x, 0, 255).astype('uint8')


def get_vgg_model(content_layers, style_layers):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
    model = tf.keras.Model(vgg.input, outputs, name="VGG_Feature_Extractor")
    return model

def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    shape = tf.shape(tensor)
    num_locations = tf.cast(shape[1] * shape[2], tf.float32)
    return result / num_locations

def compute_style_content_loss(vgg_model, loss_weights, generated_image, content_features, style_features, num_content_layers):
    content_weight, style_weight, tv_weight = loss_weights
    model_outputs = vgg_model(generated_image)
    gen_content_output = model_outputs[:num_content_layers]
    gen_style_output = model_outputs[num_content_layers:]

    content_loss = tf.add_n([tf.reduce_mean(tf.square(gen - target))
                             for gen, target in zip(gen_content_output, content_features)])
    content_loss *= content_weight

    style_loss = tf.add_n([tf.reduce_mean(tf.square(gram_matrix(gen) - gram_matrix(target)))
                           for gen, target in zip(gen_style_output, style_features)])
    style_loss *= style_weight / float(len(style_features))

    tv_loss = tf.image.total_variation(generated_image)
    tv_loss = tf.reduce_sum(tv_loss) * tv_weight

    total_loss = content_loss + style_loss + tv_loss
    return total_loss, content_loss, style_loss, tv_loss

def get_feature_representations(vgg_model, content_image, style_image, num_content_layers):
    content_outputs = vgg_model(content_image)
    style_outputs = vgg_model(style_image)
    content_features = content_outputs[:num_content_layers]
    style_features = style_outputs[num_content_layers:]
    return content_features, style_features

def teacher_train_step(vgg_model, optimizer, loss_weights, content_features, style_features, num_content_layers, generated_image):
    with tf.GradientTape() as tape:
        loss, _, _, _ = compute_style_content_loss(vgg_model, loss_weights,
                                                     generated_image, content_features,
                                                     style_features, num_content_layers)
    grad = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    # Clip values: VGG preprocess shifts range, so clip around that range
    generated_image.assign(tf.clip_by_value(generated_image, -123.68, 151.061)) # Approx range after VGG preprocessing

# --- Inference Class Definition ---

class StyleTransferInference:
    def __init__(self,
                 student_model_path,
                 target_size=(320, 320),
                 teacher_content_weight=1e4,
                 teacher_style_weight=1e3,
                 teacher_tv_weight=5):
        try:
            self.student_model = tf.keras.models.load_model(student_model_path)
            print(f"Loaded student model from {student_model_path}")
            self.student_model_size = self._get_model_size(student_model_path)
        except Exception as e:
            print(f"Error loading student model: {e}")
            # Optionally re-raise or handle gracefully
            raise ValueError(f"Error loading student model: {e}")

        self.target_size = target_size
        self.teacher_content_weight = teacher_content_weight
        self.teacher_style_weight = teacher_style_weight
        self.teacher_tv_weight = teacher_tv_weight

        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.num_content_layers = len(self.content_layers)

    def _get_model_size(self, model_path):
        try:
            size_bytes = os.path.getsize(model_path)
            return round(size_bytes / (1024 * 1024), 2)  # in MB
        except OSError:
            return "N/A" # Handle case where file might not exist during init check

    def _load_and_process(self, image_path):
        return load_and_process_image(image_path, self.target_size)

    def _deprocess(self, processed_img):
        return deprocess_image(processed_img)

    def _build_teacher_vgg(self):
        return get_vgg_model(self.content_layers, self.style_layers)

    def run_teacher(self, content_image_path, style_image_path, iterations=10):
        content_image = self._load_and_process(content_image_path)
        style_image = self._load_and_process(style_image_path)

        if content_image is None or style_image is None:
            raise ValueError("Error loading content or style images for teacher inference.")

        vgg_model = self._build_teacher_vgg()
        content_features, style_features = get_feature_representations(vgg_model, content_image, style_image, self.num_content_layers)

        generated_image = tf.Variable(content_image, dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)
        loss_weights = (self.teacher_content_weight, self.teacher_style_weight, self.teacher_tv_weight)

        start_time = time.time()
        last_loss = float('inf')
        for i in range(iterations):
            teacher_train_step(vgg_model, optimizer, loss_weights,
                               content_features, style_features, self.num_content_layers, generated_image)
            if i % 50 == 0: # Log progress occasionally
                 loss, _, _, _ = compute_style_content_loss(vgg_model, loss_weights, generated_image, content_features, style_features, self.num_content_layers)
                 print(f"Teacher Iteration {i}, Loss: {loss.numpy()}")
                 last_loss = loss.numpy()


        teacher_time = time.time() - start_time

        final_teacher_image = self._deprocess(generated_image.numpy())
        print(f"Teacher inference completed in {teacher_time:.2f} seconds.")
        return final_teacher_image, teacher_time, last_loss

    def run_student(self, content_image_path, style_image_path, iterations=1):
        # Note: Assumes student model takes BOTH content and style as input
        # based on Infrencefinal.py's predict call. Adjust if your model only takes content.
        content_image = self._load_and_process(content_image_path)
        style_image = self._load_and_process(style_image_path)

        if content_image is None or style_image is None:
            raise ValueError("Error loading content or style images for student inference.")

        start_time = time.time()
        current_content = content_image

        for i in range(iterations):
            print(f"Running student iteration {i+1}/{iterations}")
            # IMPORTANT: Ensure your student model expects a list of two inputs if using this predict call
            student_output = self.student_model.predict([current_content, style_image], verbose=0)
            current_content = student_output # Feed output back as input for next iteration

        student_time = time.time() - start_time

        # Calculate stats on the *final* output tensor before deprocessing
        stats = {
            "min": float(np.min(student_output)),
            "max": float(np.max(student_output)),
            "mean": float(np.mean(student_output)),
            "std": float(np.std(student_output))
        }

        student_output_image = self._deprocess(student_output)
        print(f"Student inference (with {iterations} iteration(s)) completed in {student_time:.2f} seconds.")
        print(f"Student output stats: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        return student_output_image, student_time, stats