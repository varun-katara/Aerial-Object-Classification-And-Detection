

import streamlit as st
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image


# PAGE CONFIG

st.set_page_config(
    page_title="Aerial Object Classifier",
    page_icon="🛩️",
    layout="centered"
)

st.title("Aerial Object Classification System")
st.write("Upload an image to classify whether it is a **Bird** or a **Drone**.")


# LOAD BEST MODEL

@st.cache_resource
def load_best_model():

    best_model_df = pd.read_csv(
        "results/best_model.csv"
    )

    model_name = best_model_df["Model"][0]

    model_map = {
        "Custom CNN": "models/custom_cnn_model.keras",
        "ResNet50": "models/resnet50_model.keras",
        "MobileNet": "models/mobilenet_model.keras",
        "EfficientNetB0": "models/efficientnetb0_model.keras",
    }

    model_path = model_map.get(model_name)

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = load_model(model_path)

    return model, model_name


model, model_name = load_best_model()

st.success(f"Loaded Best Model: {model_name}")


# CLASS NAMES

class_names = [
    "bird",
    "drone"
]

IMG_SIZE = (224, 224)


# IMAGE PREPROCESSING

def preprocess_image(uploaded_file):

    img = Image.open(uploaded_file)

    img = img.resize(IMG_SIZE)

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array, img


# FILE UPLOAD UI

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)



# PREDICTION

if uploaded_file is not None:

    img_array, display_img = preprocess_image(
        uploaded_file
    )

    st.image(
        display_img,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        with st.spinner("Making prediction..."):

            prediction = model.predict(
                img_array
            )

            class_index = np.argmax(
                prediction
            )

            confidence = np.max(
                prediction
            )

            predicted_class = class_names[
                class_index
            ]

            st.subheader("Prediction Result")

            st.write(
                f"Class: **{predicted_class.upper()}**"
            )

            st.write(
                f"Confidence: **{round(confidence * 100, 2)}%**"
            )

            # Progress bar visualization
            st.progress(
                int(confidence * 100)
            )



# FOOTER

st.markdown("---")

st.caption(
    "End-to-End Deep Learning Project | Streamlit Deployment"
)
