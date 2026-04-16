import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224, 224)

class_names = [
    "bird",
    "drone",
]


def preprocess_image(img_path):

    img = image.load_img(
        img_path,
        target_size=IMG_SIZE,
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


if __name__ == "__main__":

    best_model_df = pd.read_csv(
        "../results/best_model.csv"
    )

    model_name = best_model_df["Model"][0]

    model_map = {
        "Custom CNN": "../models/custom_cnn_model.keras",
        "ResNet50": "../models/resnet50_model.keras",
        "MobileNet": "../models/mobilenet_model.keras",
        "EfficientNetB0": "../models/efficientnetb0_model.keras",
    }

    model_path = model_map[model_name]

    model = load_model(model_path)

    print("Loaded model:", model_name)

    test_image = "../data/test/bird/example.jpg"

    if os.path.exists(test_image):

        img_array = preprocess_image(test_image)

        prediction = model.predict(img_array)

        class_index = np.argmax(prediction)

        confidence = np.max(prediction)

        print("Prediction:", class_names[class_index])
        print("Confidence:", round(confidence * 100, 2), "%")

    else:

        print("Test image not found")