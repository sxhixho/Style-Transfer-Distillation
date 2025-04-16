# inference_runner.py
import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image

# --- Helper Functions (Kept within the class or could be moved to utils.py) ---

def _load_and_process_image(path_to_img, target_size=(320, 320)):
    """Loads and preprocesses an image for VGG19."""
    try:
        # Load using TensorFlow Keras utils for consistency
        img = tf.keras.preprocessing.image.load_img(path_to_img, target_size=target_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        # Preprocess image for VGG19 (BGR ordering, zero-centering by ImageNet means)
        return tf.keras.applications.vgg19.preprocess_input(img)
    except Exception as e:
        print(f"Error loading image {path_to_img}: {e}")
        return None

def _deprocess_image(processed_img_np): # Renamed arg for clarity
    """Converts a VGG-preprocessed image (as NumPy array) back to a displayable RGB image."""
    # Ensure input is NumPy
    if not isinstance(processed_img_np, np.ndarray):
        # If it's a tensor, convert it. Otherwise, raise error.
        if tf.is_tensor(processed_img_np):
            processed_img_np = processed_img_np.numpy()
        else:
            raise TypeError(f"_deprocess_image expects a NumPy array or Tensor, got {type(processed_img_np)}")

    x = processed_img_np.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    if x.shape[-1] != 3:
        if x.shape[-1] == 1:
            x = np.concatenate([x, x, x], axis=-1)
        else:
             raise ValueError(f"Invalid input shape for deprocessing: {x.shape}")

    # Inverse of VGG19 preprocess
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR -> RGB
    return np.clip(x, 0, 255).astype('uint8')

def _get_vgg_model(content_layers, style_layers):
    """Creates a VGG19 model that returns intermediate outputs."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
    model = tf.keras.Model(vgg.input, outputs, name="VGG_Feature_Extractor")
    return model

def _gram_matrix(tensor):
    """Computes the Gram matrix."""
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    shape = tf.shape(tensor)
    num_locations = tf.cast(shape[1] * shape[2], tf.float32)
    return result / num_locations

def _compute_style_content_loss(vgg_model, loss_weights, generated_image, content_features, style_features, num_content_layers):
    """Computes losses for the teacher model."""
    content_weight, style_weight, tv_weight = loss_weights
    model_outputs = vgg_model(generated_image)
    gen_content_output = model_outputs[:num_content_layers]
    gen_style_output = model_outputs[num_content_layers:]

    content_loss = tf.add_n([tf.reduce_mean(tf.square(gen - target))
                             for gen, target in zip(gen_content_output, content_features)])
    content_loss *= content_weight

    style_loss = tf.add_n([tf.reduce_mean(tf.square(_gram_matrix(gen) - _gram_matrix(target)))
                           for gen, target in zip(gen_style_output, style_features)])
    style_loss *= style_weight / float(len(style_features))

    tv_loss = tf.image.total_variation(generated_image)
    tv_loss = tf.reduce_sum(tv_loss) * tv_weight

    total_loss = content_loss + style_loss + tv_loss
    return total_loss, content_loss, style_loss, tv_loss

def _get_feature_representations(vgg_model, content_image, style_image, num_content_layers):
    """Extracts VGG features."""
    content_outputs = vgg_model(content_image)
    style_outputs = vgg_model(style_image)
    content_features = content_outputs[:num_content_layers]
    style_features = style_outputs[num_content_layers:]
    return content_features, style_features

# Use tf.function for potential performance boost in optimization loop
@tf.function
def _teacher_train_step(vgg_model, optimizer, loss_weights, content_features, style_features, num_content_layers, generated_image):
    """Performs a single optimization step for the teacher."""
    with tf.GradientTape() as tape:
        loss, _, _, _ = _compute_style_content_loss(vgg_model, loss_weights,
                                                     generated_image, content_features,
                                                     style_features, num_content_layers)
    grad = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    # Clip values during optimization to keep them in a reasonable range for VGG preprocessing
    # The VGG preprocessing subtracts means around 100-120, so clipping around +/- 150 from that is reasonable.
    # Adjust clip_min/max if needed based on results.
    clip_min = -103.939 # Rough minimum after VGG preprocess_input
    clip_max = 255.0 - 103.939 # Rough maximum
    generated_image.assign(tf.clip_by_value(generated_image, clip_min, clip_max))


class InferenceRunner:
    # ... (__init__ remains the same as the previous corrected version) ...
    def __init__(self, student_model_path, target_size=(320, 320), default_teacher_iterations=50, default_student_iterations=1, teacher_content_weight=1e4, teacher_style_weight=1e3, teacher_tv_weight=5):
        self.target_size = target_size
        self.default_teacher_iterations = default_teacher_iterations
        self.default_student_iterations = default_student_iterations
        self.teacher_content_weight = teacher_content_weight
        self.teacher_style_weight = teacher_style_weight
        self.teacher_tv_weight = teacher_tv_weight
        self.student_model = None
        self.student_input_shape = None
        self.student_expects_list_input = False
        try:
            self.student_model = tf.keras.models.load_model(student_model_path)
            print(f"Loaded student model from {student_model_path}")
            self.student_input_shape = self.student_model.input_shape
            print(f"Detected student model input shape: {self.student_input_shape}")
            self.student_expects_list_input = isinstance(self.student_input_shape, list)
            if self.student_expects_list_input: print("Student model expects a list of inputs.")
            else: print("Student model expects a single tensor input.")
        except Exception as e: print(f"Error loading student model: {e}")
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.num_content_layers = len(self.content_layers)
        try:
            self.vgg_model = _get_vgg_model(self.content_layers, self.style_layers)
            print("VGG feature extractor model built for teacher.")
        except Exception as e: print(f"Error building VGG model: {e}"); self.vgg_model = None


    # --- Modified preprocess_for_student ---
    def _preprocess_for_student(self, vgg_processed_tensor):
        """
        Converts a VGG-processed *Tensor* to the format expected by the student model (e.g., [0, 1]).
        """
        if not tf.is_tensor(vgg_processed_tensor):
             raise TypeError(f"_preprocess_for_student expects a Tensor, got {type(vgg_processed_tensor)}")

        # Deprocess requires NumPy
        img_rgb_uint8 = _deprocess_image(vgg_processed_tensor.numpy()) # Call .numpy() here
        if img_rgb_uint8 is None: return None

        img = tf.convert_to_tensor(img_rgb_uint8, dtype=tf.float32)
        # Normalize to [0, 1] - Adjust if your student model expects a different range
        img = img / 255.0
        # Add batch dimension back if it was squeezed during deprocessing
        if len(img.shape) == 3:
             img = tf.expand_dims(img, axis=0)
        return img

    # --- run_teacher remains the same ---
    def run_teacher(self, content_image_path, style_image_path, iterations=None):
        if self.vgg_model is None: print("Error: VGG model not available."); return None, 0
        iterations = iterations if iterations is not None else self.default_teacher_iterations
        print(f"Running Teacher Inference for {iterations} iterations...")
        content_image = _load_and_process_image(content_image_path, self.target_size) # Tensor
        style_image = _load_and_process_image(style_image_path, self.target_size)   # Tensor
        if content_image is None or style_image is None: print("Error loading images."); return None, 0

        content_features, style_features = _get_feature_representations(self.vgg_model, content_image, style_image, self.num_content_layers)
        generated_image = tf.Variable(content_image, dtype=tf.float32) # Tensor Variable
        optimizer = tf.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)
        loss_weights = (self.teacher_content_weight, self.teacher_style_weight, self.teacher_tv_weight)

        start_time = time.time()
        for i in range(iterations):
            _teacher_train_step(self.vgg_model, optimizer, loss_weights, content_features, style_features, self.num_content_layers, generated_image)
            if (i + 1) % 50 == 0: print(f"  Teacher Iteration: {i+1}/{iterations}")
        teacher_time = time.time() - start_time

        # Pass the final tensor variable's value (as NumPy) to deprocess
        final_teacher_image = _deprocess_image(generated_image.numpy())
        print(f"Teacher inference completed in {teacher_time:.2f} seconds.")
        return final_teacher_image, teacher_time # Returns NumPy array

    # --- Modified run_student ---
    def run_student(self, content_image_path, style_image_path=None, iterations=None):
        if self.student_model is None: print("Student model not loaded."); return None, 0, None
        iterations = iterations if iterations is not None else self.default_student_iterations
        print(f"Running Student Inference for {iterations} iteration(s)...")

        # Load initial images as Tensors
        content_image_tensor = _load_and_process_image(content_image_path, self.target_size)
        if content_image_tensor is None: print("Error loading content image."); return None, 0, None

        style_image_tensor = None
        if self.student_expects_list_input:
            if style_image_path:
                style_image_tensor = _load_and_process_image(style_image_path, self.target_size)
                if style_image_tensor is None: print("Error loading style image."); return None, 0, None
            else:
                print("Error: Student expects two inputs, but no style image path provided."); return None, 0, None

        start_time = time.time()

        # Prepare initial inputs for the student using the refined helper
        # current_content_for_student now holds the tensor in the student's expected format (e.g., [0,1])
        current_content_for_student = self._preprocess_for_student(content_image_tensor)
        if current_content_for_student is None: print("Error preprocessing content for student."); return None, 0, None

        style_input_for_student = None
        if self.student_expects_list_input and style_image_tensor is not None:
            style_input_for_student = self._preprocess_for_student(style_image_tensor)
            if style_input_for_student is None: print("Error preprocessing style for student."); return None, 0, None

        student_output_raw = None # This will hold the raw output tensor/numpy from predict
        for i in range(iterations):
            print(f"  Student Iteration: {i+1}/{iterations}")
            predict_input = None
            if self.student_expects_list_input:
                if style_input_for_student is None: print("Error: Style input tensor missing."); return None, 0, None
                predict_input = [current_content_for_student, style_input_for_student]
                print(f"  Student Input: List of shapes {[t.shape for t in predict_input]}")
            else:
                predict_input = current_content_for_student
                print(f"  Student Input: Tensor shape {predict_input.shape}")

            try:
                # Keras predict usually returns NumPy, but let's handle if it's tensor
                student_output_raw = self.student_model.predict(predict_input)
                if isinstance(student_output_raw, list): # Handle unexpected list output
                    student_output_raw = student_output_raw[0]

                # Ensure it's a tensor for the next iteration's input processing
                if isinstance(student_output_raw, np.ndarray):
                    student_output_raw = tf.convert_to_tensor(student_output_raw, dtype=tf.float32)

            except Exception as e:
                print(f"Error during student model prediction: {e}"); return None, 0, None

            # Prepare the output as the input for the next iteration (keep as tensor [0,1])
            current_content_for_student = tf.clip_by_value(student_output_raw, 0.0, 1.0)

        student_time = time.time() - start_time

        if student_output_raw is None: return None, 0, None

        # --- Postprocessing Student Output ---
        # student_output_raw is likely a tensor here from the last iteration or conversion
        final_student_output_np = student_output_raw # Start with the raw output

        # Ensure it's numpy for stats and final conversion
        if tf.is_tensor(final_student_output_np):
            final_student_output_np = final_student_output_np.numpy() # Call .numpy() only if needed

        # Compute stats on the raw output *before* scaling for display
        stats = { "min": float(np.min(final_student_output_np)), "max": float(np.max(final_student_output_np)), "mean": float(np.mean(final_student_output_np)), "std": float(np.std(final_student_output_np)) }

        # Remove batch dim if present
        if len(final_student_output_np.shape) == 4: final_student_output_np = np.squeeze(final_student_output_np, axis=0)

        # Scale from [0, 1] (or model's range) to [0, 255] uint8 for display
        # Assuming output is [0, 1] range
        final_student_output_np = np.clip(final_student_output_np * 255.0, 0, 255)
        student_output_image = final_student_output_np.astype('uint8') # Final NumPy image

        print(f"Student inference ({iterations} iter) completed in {student_time:.2f} seconds.")
        print(f"Student raw output stats (last iter): min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}")

        # Ensure 3 channels (code from previous version is fine)
        if student_output_image.ndim == 2: student_output_image = np.stack([student_output_image] * 3, axis=-1)
        elif student_output_image.ndim == 3 and student_output_image.shape[-1] == 1: student_output_image = np.concatenate([student_output_image] * 3, axis=-1)
        elif student_output_image.ndim != 3 or student_output_image.shape[-1] != 3: print(f"Warning: Unexpected student output shape: {student_output_image.shape}")

        return student_output_image, student_time, stats # Returns NumPy array

    # --- Modified get_display_image ---
    def get_display_image(self, image_path):
        """Loads an image and returns a displayable NumPy array."""
        processed_tensor = _load_and_process_image(image_path, self.target_size) # Returns tensor
        if processed_tensor is None: return None
        try:
            # _deprocess_image now handles tensor input via internal .numpy() call if needed,
            # but it's cleaner to pass the expected NumPy array directly.
            return _deprocess_image(processed_tensor.numpy()) # Pass NumPy array
        except (ValueError, TypeError) as e:
            print(f"Error deprocessing image {image_path}: {e}")
            return None

    # --- save_image remains the same (already checks for NumPy uint8) ---
    def save_image(self, image_array, save_path):
        # ... (previous save_image code is fine) ...
        if image_array is None: print(f"Attempted to save a None image to {save_path}"); return False
        if not isinstance(image_array, np.ndarray) or image_array.ndim != 3 or image_array.shape[-1] != 3: print(f"Error: Attempting to save invalid image data to {save_path}. Shape: {image_array.shape if isinstance(image_array, np.ndarray) else type(image_array)}, Dtype: {image_array.dtype if isinstance(image_array, np.ndarray) else 'N/A'}"); return False
        if image_array.dtype != np.uint8:
             print(f"Warning: Image data type is {image_array.dtype}, expected uint8. Attempting conversion for saving {save_path}.")
             if np.issubdtype(image_array.dtype, np.floating) and np.max(image_array) <= 255.1 and np.min(image_array) >= -0.1: image_array = np.clip(image_array, 0, 255).astype(np.uint8)
             else: print(f"Error: Cannot reliably convert image data type {image_array.dtype} to uint8 for saving."); return False
        try:
            img = Image.fromarray(image_array)
            img.save(save_path)
            print(f"Saved image to {save_path}")
            return True
        except Exception as e: print(f"Error saving image {save_path}: {e}"); return False
        
if __name__ == '__main__':
    print("Testing InferenceRunner...")
    content_img = 'test1.jpg' # Make sure this exists
    style_img_predefined = 'static/styles/Pixel.png' # Make sure this exists
    student_model_file = 'student_model27.keras' # Make sure this exists
    output_dir = 'test_outputs'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(content_img) or not os.path.exists(style_img_predefined) or not os.path.exists(student_model_file):
         print("Error: Ensure test1.jpg, static/styles/Pixel.png, and student_model27.keras exist for testing.")
    else:
        runner = InferenceRunner(
            student_model_path=student_model_file,
            target_size=(320, 320),
            default_teacher_iterations=10 # Quick test
        )

        print("\n--- Running Teacher ---")
        teacher_img, teacher_t = runner.run_teacher(content_img, style_img_predefined, iterations=15)
        if teacher_img is not None:
            runner.save_image(teacher_img, os.path.join(output_dir, 'test_teacher_output.jpg'))

        print("\n--- Running Student ---")
        # Assuming student is single input based on original student_model.py
        # If your student_model27.keras expects two inputs, provide style_img_predefined here.
        student_img, student_t, student_stats = runner.run_student(content_img, style_image_path=None, iterations=3)
        # Example if student *did* expect two inputs:
        # student_img, student_t, student_stats = runner.run_student(content_img, style_image_path=style_img_predefined, iterations=3)

        if student_img is not None:
            runner.save_image(student_img, os.path.join(output_dir, 'test_student_output.jpg'))

        print("\n--- Getting Display Images ---")
        content_disp = runner.get_display_image(content_img)
        style_disp = runner.get_display_image(style_img_predefined)
        if content_disp is not None:
             runner.save_image(content_disp, os.path.join(output_dir, 'test_content_disp.jpg'))
        if style_disp is not None:
             runner.save_image(style_disp, os.path.join(output_dir, 'test_style_disp.jpg'))

        print("\nTest finished.")