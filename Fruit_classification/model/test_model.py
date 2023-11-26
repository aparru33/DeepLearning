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
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from tensorflow import expand_dims
import pandas as pd

print("load image")
PATH = '/home/ubuntu/workspace/finovox_main/main-repo/backend/_1_models/datastructure/zoning/logo_detection/'
FILES = PATH+'to_test/'
model1= load_model(PATH+'custom_model_201123_12h56m58.keras')
model2= load_model(PATH+'custom_model_221123_10h39m47.keras')
model3= load_model(PATH+'custom_model_221123_11h16m42.keras')
model4= load_model(PATH+'custom_model_221123_15h08m46.keras')

input_shape = lambda x :(x.layers[0].input_shape[1],x.layers[0].input_shape[2] )

model_dict = {'custom_model_201123_12h56m58.keras' :model1,'custom_model_221123_10h39m47.keras':model2,
              'custom_model_221123_11h16m42.keras' : model3,'custom_model_221123_15h08m46.keras':model4}


res=[]
for f in os.listdir(FILES):
    print("##############################################")
    print(f)
    for k,m in model_dict.items():
        print(k)
        img = load_img(FILES+f, target_size=input_shape(m))
        img_array = img_to_array(img)
        img_array = expand_dims(img_array, 0)
        pred = 1 - m.predict(img_array)
        res.append({"file":f,"model":k,"pred":pred[0][0], 'is_true_logo': 1 if 'logo' in f else 0})
        print(f"score for {f} with {k} is {pred[0][0]}")

df = pd.DataFrame(res)
df.to_csv(PATH+'res_test.csv',index=False)
# %%
