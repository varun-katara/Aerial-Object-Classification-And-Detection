import tensorflow as tf
import os

from data_preprocessing import create_data_generators

DATA_DIR = "../data"


MODELS = {
    "ResNet50": tf.keras.applications.ResNet50,
    "MobileNet": tf.keras.applications.MobileNet,
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
}


if __name__ == "__main__":

    train_gen, val_gen, test_gen = create_data_generators(DATA_DIR)

    for model_name, model_fn in MODELS.items():

        print("Training:", model_name)

        base_model = model_fn(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )

        base_model.trainable = False

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        outputs = tf.keras.layers.Dense(
            2,
            activation="softmax",
        )(x)

        model = tf.keras.Model(
            inputs=base_model.input,
            outputs=outputs,
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=5,
        )

        os.makedirs("../models", exist_ok=True)

        model.save(
            f"../models/{model_name.lower()}_model.keras"
        )