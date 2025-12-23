import logging

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Limit GPU memory growth
try:
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logger.info("GPU memory growth enabled")
except tf.errors.InvalidArgumentError as e:
    logger.error(f"Invalid argument error when setting GPU memory growth: {e}")
except tf.errors.ResourceExhaustedError as e:
    logger.error(
        f"Resource exhausted error (insufficient memory) when setting GPU memory growth: {e}"
    )
except Exception as e:  # pylint: disable=broad-except
    logger.error(f"Failed to set GPU memory growth: {e}")

# Load pre-trained I3D model
i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures["default"]  # type: ignore


# Define LSTM model
def build_lstm_model(num_classes, feature_dim=400):
    input_shape = (None, feature_dim)
    inputs = Input(shape=input_shape)
    x = LSTM(128)(inputs)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
