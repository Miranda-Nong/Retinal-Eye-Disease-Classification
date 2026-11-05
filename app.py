
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="Eye Disease Classification",
    page_icon="👁️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("resnet50_eye_model.keras")

model = load_model()

# Keep this order consistent with the class_names
# printed by your original training notebook.
class_names = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

st.title("👁️ Eye Disease Classification")
st.write(
    "Upload a retinal or fundus image to obtain a preliminary "
    "model prediction. This tool does not replace an eye examination."
)

uploaded_file = st.file_uploader(
    "Upload an eye image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            resized_image = image.resize((256, 256))
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            # Match the preprocessing used during model training.
            image_array = preprocess_input(image_array)

            predictions = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        st.subheader("Prediction")
        st.success(f"Predicted class: {predicted_class.replace('_', ' ').title()}")
        st.write(f"Model confidence: {confidence:.2f}%")

        st.warning(
            "This is an experimental research prediction, not a medical "
            "diagnosis. Please consult an eye-care professional for assessment."
        )
