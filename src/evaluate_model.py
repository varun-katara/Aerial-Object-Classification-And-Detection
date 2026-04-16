import pandas as pd
import os
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

from data_preprocessing import create_data_generators

DATA_DIR = "../data"
MODEL_PATH = "../models/custom_cnn_model.keras"


if __name__ == "__main__":

    train_gen, val_gen, test_gen = create_data_generators(DATA_DIR)

    model = load_model(MODEL_PATH)

    predictions = model.predict(test_gen)

    y_pred = predictions.argmax(axis=1)

    class_names = list(test_gen.class_indices.keys())

    report = classification_report(
        test_gen.classes,
        y_pred,
        target_names=class_names,
        output_dict=True,
    )

    df = pd.DataFrame(report)

    os.makedirs("../results", exist_ok=True)

    df.to_csv("../results/result.csv")

    print("Evaluation saved to results/result.csv")