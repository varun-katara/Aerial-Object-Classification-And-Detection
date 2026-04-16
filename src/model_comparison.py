import pandas as pd
import os


if __name__ == "__main__":

    results = {
        "Model": [
            "Custom CNN",
            "ResNet50",
            "MobileNet",
            "EfficientNetB0",
        ],
        "Accuracy": [
            0.910224,
            0.710723,
            0.982544,
            0.531172,
        ],
        "Precision": [
            0.912029,
            0.715865,
            0.982550,
            0.282144,
        ],
        "Recall": [
            0.910224,
            0.710723,
            0.982544,
            0.531172,
        ],
        "F1-score": [
            0.910312,
            0.705920,
            0.982541,
            0.368533,
        ],
    }

    comparison_df = pd.DataFrame(results)

    os.makedirs("../results", exist_ok=True)

    comparison_df.to_csv(
        "../results/model_comparison.csv",
        index=False,
    )

    best_model = comparison_df.sort_values(
        by="Accuracy",
        ascending=False,
    ).head(1)

    best_model.to_csv(
        "../results/best_model.csv",
        index=False,
    )

    print("Best model selected:")
    print(best_model)