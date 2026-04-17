# Aerial Object Classification and Detection (Bird vs Drone)

## Project Overview

This project is an **end-to-end Deep Learning pipeline** that classifies aerial images as either **Bird** or **Drone** using Convolutional Neural Networks (CNN) and Transfer Learning models.

The system covers the complete machine learning lifecycle:

* Data inspection and preprocessing
* Model training and evaluation
* Transfer learning implementation
* Model comparison and selection
* Model saving and inference pipeline
* Web application deployment using Streamlit

The final model is deployed as an interactive web application where users can upload an image and receive a prediction along with a confidence score.

---

## Live Application

The application is deployed using **Streamlit Cloud**.

**Access the deployed app here:**

```text
Paste your Streamlit link here
Example:
https://aerial-object-classification-and-detection-vqpbadih782t9drjpmu.streamlit.app/
```

---

## Features

* Image classification (Bird vs Drone)
* Transfer Learning models:

  * Custom CNN
  * ResNet50
  * MobileNet
  * EfficientNetB0
* Model performance comparison
* Automatic best model selection
* Real-time prediction interface
* Confidence score display
* End-to-end ML pipeline
* Production-ready project structure

---

## Project Workflow

```text
Data Collection
      ↓
Data Inspection
      ↓
Data Preprocessing
      ↓
Train / Validation Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Transfer Learning
      ↓
Model Comparison
      ↓
Best Model Selection
      ↓
Model Saving
      ↓
Inference Pipeline
      ↓
Streamlit Deployment
```

---

## Model Performance (Example)

| Model          | Accuracy  | Precision | Recall    | F1-score  |
| -------------- | --------- | --------- | --------- | --------- |
| Custom CNN     | 0.910     | 0.912     | 0.910     | 0.910     |
| ResNet50       | 0.711     | 0.716     | 0.711     | 0.706     |
| MobileNet      | **0.983** | **0.983** | **0.983** | **0.983** |
| EfficientNetB0 | 0.531     | 0.282     | 0.531     | 0.369     |

**Selected Best Model:** MobileNet

---

## Tech Stack

### Programming Language

* Python

### Libraries and Frameworks

* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Pillow
* Streamlit

### Tools

* Jupyter Notebook
* Git
* GitHub
* Streamlit Cloud

---

## Project Structure

```text
Aerial-Object-Classification-and-Detection/

├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
├── .python-version
├── .gitignore
├── .gitattributes

├── .devcontainer/
│   └── devcontainer.json

├── models/
│   └── mobilenet_model.keras

├── results/
│   └── best_model.csv

├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── transfer_learning_models.py
│   ├── model_comparison.py
│   ├── inference_pipeline.py
│   └── main.py
```

---

## Installation and Setup (Local)

### Step 1 — Clone Repository

```bash
git clone https://github.com/varun-katara/Aerial-Object-Classification-And-Detection
cd Aerial-Object-Classification-And-Detection
```

### Step 2 — Create Virtual Environment

```bash
python -m venv .venv
```

### Step 3 — Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / Mac:

```bash
source .venv/bin/activate
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Run Application

```bash
streamlit run app.py
```

---

## How the Model Works

1. User uploads an image
2. Image is resized and normalized
3. Model predicts the class
4. System returns:

* Predicted label (Bird / Drone)
* Confidence score

---

## Example Output

```text
Prediction: DRONE
Confidence: 98.25%
```

---

## Future Improvements

* Object detection using YOLO
* Multi-class aerial object classification
* Model monitoring and logging
* Docker containerization
* Cloud deployment using AWS
* CI/CD pipeline integration

---

## Author

Varun Katara

Mechanical Engineer transitioning into IT and Machine Learning
Experience in:

* Python Development
* Linux System Administration
* Machine Learning Projects
* Data Analysis

---

## License

This project is licensed under the MIT License.

---

## Acknowledgment

This project was developed as part of hands-on learning in:

* Deep Learning
* Model Deployment
* End-to-End Machine Learning Pipelines
* Production-ready ML Systems
