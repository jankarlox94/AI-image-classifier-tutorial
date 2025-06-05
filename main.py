import numpy as np
import streamlit as st

from PIL import Image

import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def load_model():
    model = ResNet50(input_shape = (224,224,3), weights='imagenet')
    return model

def preprocess_image(uploaded_img):
    img_path = uploaded_img
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def classify_image(model, uploaded_img):
    try:
        processed_image = preprocess_image(uploaded_img)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None
    
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")
    
    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what is in it!")
    
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()
    
    uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_img is not None:
        image = st.image(
            uploaded_img, caption="Uploaded Image", use_container_width=True
        )
        btn = st.button("Classify Image")
        
        if btn:
            with st.spinner("Analyzing Image..."):
               
                predictions = classify_image(model, uploaded_img)
                
                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions: 
                        st.write(f"**{label}**: {score:.2%}")
                        
                        
if __name__ == "__main__":
    main()