from rembg import remove 
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# Load your trained model
model = tf.keras.models.load_model("model.h5")

def remove_background(input_path):
     # Processing the image 
    # Removing the background from the given Image 
    return remove(Image.open(input_path)) 


def classify_image(image):
    return 0
    # # Preprocess the image for the model
    # image = image.resize((64, 64))
    # img_array = np.array(image) / 255.0
    # img_array = np.expand_dims(img_array, axis=0)

    # # Get the prediction and return the class
    # prediction = model.predict(img_array)
    # class_id = np.argmax(prediction)
    # return class_id 

def main():
    st.title("Fruit Image Classification")

    st.sidebar.title("Upload an image of fruit")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = remove_background(uploaded_file)
        st.image(image, caption='Uploaded Fruit Image.', use_column_width=True)
        st.write("Predicting...")
        label = classify_image(image)
        st.write(f'Prediction: {label}')


if __name__ == "__main__":
    main()
