import tensorflow as tf
import numpy as np
import os, glob, random

# Global configuration for image dimensions
IMG_HEIGHT = 320
IMG_WIDTH = 320
TARGET_SIZE = (IMG_HEIGHT, IMG_WIDTH)

class DataProcessor:
    """
    Handles image processing, loading, and dataset creation.
    """
    def __init__(self, target_size=TARGET_SIZE):
        self.target_size = target_size

    def load_and_process_image(self, path_to_img):
        try:
            img = tf.keras.preprocessing.image.load_img(path_to_img, target_size=self.target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            # Preprocessing for VGG19
            return tf.keras.applications.vgg19.preprocess_input(img)
        except Exception as e:
            print(f"Error loading image {path_to_img}: {e}")
            return None

    def deprocess_image(self, processed_img):
        x = processed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        # Undo VGG preprocessing
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]  # Convert BGR to RGB
        return np.clip(x, 0, 255).astype('uint8')

    def load_triplet(self, content_path, style_path, target_path):
        content_path_str = content_path.numpy().decode('utf-8')
        style_path_str = style_path.numpy().decode('utf-8')
        target_path_str = target_path.numpy().decode('utf-8')

        content_img = self.load_and_process_image(content_path_str)
        style_img = self.load_and_process_image(style_path_str)
        target_img = self.load_and_process_image(target_path_str)

        if content_img is None or style_img is None or target_img is None:
            print("Warning: Error loading one of the images.")
            dummy_shape = (1, IMG_HEIGHT, IMG_WIDTH, 3)
            content_img = np.zeros(dummy_shape, dtype=np.float32)
            style_img = np.zeros(dummy_shape, dtype=np.float32)
            target_img = np.zeros(dummy_shape, dtype=np.float32)

        content_img = tf.squeeze(content_img, axis=0)
        style_img = tf.squeeze(style_img, axis=0)
        target_img = tf.squeeze(target_img, axis=0)
        return content_img, style_img, target_img

    def load_triplet_tf(self, content_path, style_path, target_path):
        result_signature = (
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)
        )
        content_img, style_img, target_img = tf.py_function(
            func=self.load_triplet,
            inp=[content_path, style_path, target_path],
            Tout=result_signature
        )
        content_img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
        style_img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
        target_img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
        return (content_img, style_img), target_img

    def create_dataset(self, content_dir, style_dir, teacher_output_dir, batch_size, validation_split=0.1):
        content_paths = glob.glob(os.path.join(content_dir, '*.jpg')) + glob.glob(os.path.join(content_dir, '*.png'))
        style_paths = glob.glob(os.path.join(style_dir, '*.jpg')) + glob.glob(os.path.join(style_dir, '*.png'))

        triplets = []
        for c_path in content_paths:
            for s_path in style_paths:
                c_name = os.path.splitext(os.path.basename(c_path))[0]
                s_name = os.path.splitext(os.path.basename(s_path))[0]
                target_filename = f"{c_name}_style_{s_name}.png"
                target_filepath = os.path.join(teacher_output_dir, target_filename)
                if os.path.exists(target_filepath):
                    triplets.append((c_path, s_path, target_filepath))

        if not triplets:
            raise ValueError("No matching teacher outputs found. Did you run the generation step?")
        random.shuffle(triplets)
        num_val_samples = int(len(triplets) * validation_split)
        train_triplets = triplets[num_val_samples:]
        val_triplets = triplets[:num_val_samples]

        def build_dataset(triplet_list):
            if not triplet_list:
                return tf.data.Dataset.from_tensor_slices(([], [], [])).map(
                    self.load_triplet_tf, num_parallel_calls=tf.data.AUTOTUNE)
            content_list, style_list, target_list = zip(*triplet_list)
            ds = tf.data.Dataset.from_tensor_slices((list(content_list), list(style_list), list(target_list)))
            ds = ds.map(self.load_triplet_tf, num_parallel_calls=tf.data.AUTOTUNE)
            return ds

        train_ds = build_dataset(train_triplets)
        val_ds = build_dataset(val_triplets)
        train_ds = train_ds.shuffle(buffer_size=len(train_triplets)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        if val_triplets:
            val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train_ds, val_ds

if __name__ == '__main__':
    # Simple test/demo code for DataProcessor
    dp = DataProcessor()
    print("DataProcessor loaded. Test image loading here if needed.")
