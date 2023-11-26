import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
import pandas as pd

# Load your trained model
PATH_MODEL ="/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/model/saved_models/resnest_model_2023-11-24 23:08:27.409031.keras"

cat_name = ['apple', 'apricot', 'banana', 'barberry', 'black_berry', 'black_cherry', 'brazil_nut', 'cashew', 'cherry', 'clementine', 'coconut', 'dragonfruit', 'durian', 'fig', 'grapefruit', 'jujube', 'kiwi', 'lime', 'mango', 'olive', 'orange', 'papaya', 'passion_fruit', 'pineapple', 'pomegranate', 'raspberry', 'red_mulberry', 'strawberry', 'tomato', 'watermelon', 'yuzu'] 

              
model = tf.keras.models.load_model(PATH_MODEL)

def resize_and_pad(img_path, target_size=(224, 224), fill_color=(0, 0, 0)):

    img= Image.open(img_path)
    if img.mode == 'P':
        img = img.convert('RGBA')

    # Calculate the ratio to resize the image
    ratio = min(target_size[0] / img.size[0], target_size[1] / img.size[1])
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))

    # Resize the image
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert RGBA images to RGB to avoid issues with JPEG format
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, fill_color)
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background

    # Create a new image and paste the resized image onto the center
    new_img = Image.new('RGB', target_size, fill_color)
    new_img.paste(img, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    return new_img


def classify_image(_uploaded_image):
    # # Preprocess the image for the model

    img = resize_and_pad(_uploaded_image)
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)
    pred = model.predict(img_array)[0]
    return cat_name[np.argmax(pred[0])]

def main():
    st.title("Fruit Image Classification")
    st.write("This is a simple image classification web app to predict fruit name.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = resize_and_pad(uploaded_file)
        st.image(image, caption='Uploaded Fruit Image.', use_column_width=True)
        st.write("Predicting...")
        label = classify_image(image)
        st.write(f'Prediction: {label}')


if __name__ == "__main__":
    main()
