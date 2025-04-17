import tensorflow as tf
import numpy as np
import time, os, glob
from PIL import Image
from DataProcessor import DataProcessor

# Global configuration values for teacher process
TEACHER_ITERATIONS = 10
TEACHER_CONTENT_WEIGHT = 1e4
TEACHER_STYLE_WEIGHT = 1e3
TEACHER_TV_WEIGHT = 5
IMG_HEIGHT = 320
IMG_WIDTH = 320

class ModelTrainer:
    """
    Handles teacher style transfer generation and student model training.
    """
    def __init__(self, data_processor=None):
        self.data_processor = data_processor if data_processor else DataProcessor()

    @staticmethod
    def get_vgg_model(content_layers, style_layers):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in content_layers + style_layers]
        return tf.keras.Model(vgg.input, outputs, name="VGG_Feature_Extractor")

    @staticmethod
    def gram_matrix(tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
        num_locations = tf.cast(tf.shape(tensor)[1] * tf.shape(tensor)[2], tf.float32)
        return result / num_locations

    @staticmethod
    def compute_style_content_loss(vgg_model, loss_weights, generated_image, content_features, style_features, num_content_layers):
        content_weight, style_weight, tv_weight = loss_weights
        outputs = vgg_model(generated_image)
        gen_content = outputs[:num_content_layers]
        gen_style = outputs[num_content_layers:]

        content_loss = tf.add_n([tf.reduce_mean(tf.square(gen - target))
                                 for gen, target in zip(gen_content, content_features)]) * content_weight
        style_loss = tf.add_n([tf.reduce_mean(tf.square(ModelTrainer.gram_matrix(gen) - ModelTrainer.gram_matrix(target)))
                               for gen, target in zip(gen_style, style_features)]) * (style_weight / len(style_features))
        tv_loss = tf.reduce_sum(tf.image.total_variation(generated_image)) * tv_weight
        total_loss = content_loss + style_loss + tv_loss
        return total_loss, content_loss, style_loss, tv_loss

    @staticmethod
    @tf.function
    def teacher_train_step(vgg_model, optimizer, loss_weights, content_features, style_features, num_content_layers, generated_image):
        with tf.GradientTape() as tape:
            loss, _, _, _ = ModelTrainer.compute_style_content_loss(
                vgg_model, loss_weights, generated_image, content_features, style_features, num_content_layers)
        grad = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, -120, 120))

    def get_feature_representations(self, vgg_model, content_image, style_image, num_content_layers):
        content_outputs = vgg_model(content_image)
        style_outputs = vgg_model(style_image)
        return content_outputs[:num_content_layers], style_outputs[num_content_layers:]

    def run_style_transfer_teacher(self, content_path, style_path, output_path,
                                   num_iterations=TEACHER_ITERATIONS,
                                   content_weight=TEACHER_CONTENT_WEIGHT,
                                   style_weight=TEACHER_STYLE_WEIGHT,
                                   tv_weight=TEACHER_TV_WEIGHT):
        print(f"Generating teacher output for {os.path.basename(content_path)} + {os.path.basename(style_path)}")
        content_image = self.data_processor.load_and_process_image(content_path)
        style_image = self.data_processor.load_and_process_image(style_path)
        if content_image is None or style_image is None:
            print("Skipping due to loading error.")
            return

        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        num_content_layers = len(content_layers)
        vgg_model = ModelTrainer.get_vgg_model(content_layers, style_layers)

        content_features, style_features = self.get_feature_representations(vgg_model, content_image, style_image, num_content_layers)
        generated_image = tf.Variable(content_image, dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)
        loss_weights = (content_weight, style_weight, tv_weight)
        start_time = time.time()
        for i in range(num_iterations):
            ModelTrainer.teacher_train_step(vgg_model, optimizer, loss_weights,
                                            content_features, style_features, num_content_layers, generated_image)
        print(f"Generation took {time.time()-start_time:.1f} sec")
        final_image = self.data_processor.deprocess_image(generated_image.numpy())
        try:
            Image.fromarray(final_image).save(output_path)
            print(f"Saved teacher output to {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def generate_teacher_outputs(self, content_dir, style_dir, output_dir):
        print("Starting teacher output generation...")
        content_paths = glob.glob(os.path.join(content_dir, '*.jpg')) + glob.glob(os.path.join(content_dir, '*.png'))
        style_paths = glob.glob(os.path.join(style_dir, '*.jpg')) + glob.glob(os.path.join(style_dir, '*.png'))
        if not content_paths or not style_paths:
            print("No content or style images found!")
            return
        total_pairs = len(content_paths) * len(style_paths)
        generated_count = 0
        for c_path in content_paths:
            for s_path in style_paths:
                c_name = os.path.splitext(os.path.basename(c_path))[0]
                s_name = os.path.splitext(os.path.basename(s_path))[0]
                out_filename = f"{c_name}_style_{s_name}.png"
                output_filepath = os.path.join(output_dir, out_filename)
                if os.path.exists(output_filepath):
                    continue
                self.run_style_transfer_teacher(c_path, s_path, output_filepath)
                generated_count += 1
                print(f"Progress: {generated_count}/{total_pairs}")
        print("Teacher data generation complete.")

    @staticmethod
    def build_student_model(height=IMG_HEIGHT, width=IMG_WIDTH):
        """ Builds a simple U-Net like student model. """
        # Inputs for content and style images
        content_input = tf.keras.layers.Input(shape=(height, width, 3), name="content_input")
        style_input = tf.keras.layers.Input(shape=(height, width, 3), name="style_input")

        # Concatenate inputs along the channel axis
        x = tf.keras.layers.Concatenate(axis=-1)([content_input, style_input]) # Shape: (h, w, 6)

        # Simple Encoder-Decoder Structure (adjust complexity as needed)
        # Encoder
        e1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        e1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(e1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(e1) # h/2, w/2

        e2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(p1)
        e2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(e2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(e2) # h/4, w/4

        # Bottleneck
        b = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(p2)
        b = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(b)

        # Decoder
        u1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b) # h/2, w/2
        # Skip connection might need cropping/padding if sizes mismatch, but 'same' padding helps
        # Ensure channel counts match before concat if adding skip connections like U-Net
        # Simple version without skip connections for now:
        d1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(u1)
        d1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(d1)


        u2 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1) # h, w
        d2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(u2)
        d2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(d2)

        # Output layer - 3 channels (RGB)
        # No activation here - output raw values comparable to VGG input range
        # Activation (like tanh or sigmoid) could be added if targets are normalized to [-1,1] or [0,1]
        output_image = tf.keras.layers.Conv2D(3, (1, 1), padding='same', name="stylized_output")(d2)

        model = tf.keras.Model(inputs=[content_input, style_input], outputs=output_image, name="Student_Stylizer")
        return model


    def train_student_model(self, student_model, train_dataset, val_dataset, epochs, learning_rate, checkpoint_dir):
        print("Starting student model training...")
        loss_fn = tf.keras.losses.MeanAbsoluteError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=student_model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f"Restored checkpoint from {ckpt_manager.latest_checkpoint}")
        best_val_loss = float('inf')
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for inputs, targets in train_dataset:
                content_batch, style_batch = inputs
                with tf.GradientTape() as tape:
                    preds = student_model([content_batch, style_batch], training=True)
                    loss = loss_fn(targets, preds)
                grads = tape.gradient(loss, student_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
            # Validation loop
            val_loss = tf.keras.metrics.Mean()
            for inputs_val, targets_val in val_dataset:
                c_val, s_val = inputs_val
                preds_val = student_model([c_val, s_val], training=False)
                val_loss.update_state(loss_fn(targets_val, preds_val))
            print(f"Validation Loss: {val_loss.result():.4f}")
            if val_loss.result() < best_val_loss:
                best_val_loss = val_loss.result()
                ckpt_manager.save()
                print("Checkpoint saved.")
        print("Student model training complete.")

if __name__ == '__main__':
    mt = ModelTrainer()
    print("ModelTrainer loaded. Run tests as needed.")
