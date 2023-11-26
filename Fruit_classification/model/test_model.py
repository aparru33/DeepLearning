#%%
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from datetime import datetime
from sklearn.metrics import classification_report

# Set variables
BATCH_SIZE = 256
PROJECT_PATH = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification'
MODEL_DIR = os.path.join(PROJECT_PATH, 'model/saved_models')
TEST_DIR = os.path.join(PROJECT_PATH, 'dataset/final_dataset/test')


def write_metrics_model(y_pred, y_true, log, name_category):
    report = classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in name_category], zero_division=0)
    log.write(report)

def get_metrics_model(model_path, log):
    try:
        log.write(f'Get metrics for model {model_path} \n')
        model = load_model(model_path)
        input_shape = (224, 224)

        test_dataset = image_dataset_from_directory(
            TEST_DIR,
            image_size=input_shape,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=42,
        )

        print(test_dataset.class_names)

        y_true, y_pred = [], []
        for img_batch, label_batch in test_dataset:
            y_true_batch = label_batch.numpy()
            y_true.extend(y_true_batch)

            pred_probs = model.predict(img_batch)
            pred_classes = np.argmax(pred_probs, axis=1)
            y_pred.extend(pred_classes)

        y_true = np.array(y_true)

        # Adjust if y_true is one-hot encoded
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        y_pred = np.array(y_pred)
        write_metrics_model(y_pred, y_true, log,test_dataset.class_names)

    except Exception as e:
        log.write(f"Error in get_metrics_model: {str(e)}\n")



# Create log file and start test
log_file_path = os.path.join(PROJECT_PATH, f"model/test_log/log_test_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
with open(log_file_path, 'w', encoding='utf-8') as log:
    log.write("Start test \n")
    log.write("Get true labels and predicted labels \n")

    models = [os.path.join(MODEL_DIR, m) for m in os.listdir(MODEL_DIR) if m.endswith('.keras') or m.endswith('.h5')]
    for model_path in models:
        get_metrics_model(model_path, log)

# %%
# test on new data
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
import pandas as pd


from PIL import Image

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

def find_top_three_indices(lst):
    return np.argsort(np.array(lst))[-3:]

print("load image")
#PATH = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/dataset/Fruits_360/test-multiple_fruits/'
PATH = "/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/dataset/images/images/"
# Create log file and start test
log_file_path = os.path.join(PROJECT_PATH, f"model/test_log/log_test_model_on_new_data_one_category{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
with open(log_file_path, 'w', encoding='utf-8') as log:
    log.write("Start test \n")
    log.write("Get true labels and predicted labels \n")

    model_path = [os.path.join(MODEL_DIR, m) for m in os.listdir(MODEL_DIR) if m.endswith('.keras') or m.endswith('.h5')]
    list_model =[]
    for mp in model_path:
        list_model.append({'name' : mp.split('/')[-1], 'model' : load_model(mp)})
    cat_name =['apple', 'apricot', 'banana', 'barberry', 'black_berry', 'black_cherry', 'brazil_nut', 'cashew', 'cherry', 'clementine', 'coconut', 'dragonfruit', 'durian', 'fig', 'grapefruit', 'jujube', 'kiwi', 'lime', 'mango', 'olive', 'orange', 'papaya', 'passion_fruit', 'pineapple', 'pomegranate', 'raspberry', 'red_mulberry', 'strawberry', 'tomato', 'watermelon', 'yuzu']
    log.write(f'list of categpry : {cat_name} \n')
    for d in os.listdir(PATH):
        log.write("category is "+d +'\n')
        if os.path.isdir(PATH+d):
            for f in os.listdir(PATH+d):
                print("##############################################")
                log.write(f"file is {f} \n")
                for m in list_model:
                    img = resize_and_pad(PATH+d+'/'+f)
                    img_array = img_to_array(img)
                    img_array = expand_dims(img_array, 0)
                    start_time = datetime.now()
                    pred = m['model'].predict(img_array)[0]
                    end_time = datetime.now()
                    log.write(f"prediction time is {(end_time-start_time).microseconds} microseconds \n")
                    #print(pred)
                    #print(top)
                    #print((f"{cat_name[np.argmax(pred[0])]}"))
                    if cat_name[np.argmax(pred)] == d:
                        log.write(f"{m['name']} is true : {d} \n" )
                    log.write("\n")

# %%
