# models/student_model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_student_model(input_shape=(512, 512, 3)):
    """Builds a compact CNN-based student model for fast style transfer."""
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)

    # Bottleneck
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)

    # Decoder
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)

    # Output layer (3 channels, pixel values in [0,1])
    outputs = layers.Conv2D(3, kernel_size=1, padding='same', activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="StudentStyleTransfer")
    return model


def load_trained_model(weights_path="models/student_model.weights.h5", input_shape=(512, 512, 3)):
    """Loads the student model with pretrained weights."""
    model = build_student_model(input_shape=input_shape)
    model.load_weights(weights_path)
    return model
