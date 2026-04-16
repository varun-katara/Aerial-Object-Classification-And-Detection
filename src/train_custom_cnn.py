import tensorflow as tf
import time
import os

from data_preprocessing import create_data_generators

DATA_DIR = "../data"


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=2):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

if __name__ == "__main__":

    train_gen, val_gen, test_gen = create_data_generators(DATA_DIR)

    model = build_custom_cnn()

    start_time = time.time()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
    )

    end_time = time.time()

    training_time_minutes = round((end_time - start_time) / 60, 2)

    os.makedirs("../models", exist_ok=True)

    model.save("../models/custom_cnn_model.keras")

    print("Training completed")
    print("Training Time (min):", training_time_minutes)