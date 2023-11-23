from keras.preprocessing import image_dataset_from_directory
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization,Rescaling
from keras.layers import LeakyReLU
from keras import models
from keras.metrics import Precision, Recall, AUC
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

from datetime import datetime
import numpy as np

def create_model(input_shape, num_classes):
    model = models.Sequential()

    # Input Layer
    model.add(Input(shape=input_shape))

    # Normalization Layer
    model.add(Rescaling(1./255))

    # First Convolutional Block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))

    # Max Pooling Layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout Layer
    model.add(Dropout(0.25))

    # Flatten Layer
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the Model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', Precision(), Recall(), AUC()])

    return model



def create_model_vg(_input_shape):
    print("Create the vgg16 model \n")
    vgg16_model = models.Sequential([
        Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu", input_shape=_input_shape),
        Conv2D(filters=32, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        
        Conv2D(filters=356, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=356, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=356, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        
        Conv2D(filters=356, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=356, kernel_size=(3,3), padding="same", activation="relu"),
        Conv2D(filters=356, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(1024),
        LeakyReLU(alpha=0.01),
        Dense(1024),
        LeakyReLU(alpha=0.01),
        Dense(num_classes, activation="softmax")  # For binary classification
    ])

# Set the path to your dataset
TRAIN_DIR = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/dataset/train'
VALIDATION_DIR = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/dataset/validation'

# Bilinear, bicubic, lanczos5, mitchellcubic.
INTERPOLATION = "bicubic"

IMAGE_CHANNELS=3
batch_size = 256
img_height = 100
img_width = 100


try:
    log = open("/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/model/log.txt", "w")
    log.write("Load datasets\n")

    train_dataset = image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
    )
    log.write("train datasets loaded\n")
    validation_dataset = image_dataset_from_directory(
        VALIDATION_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )
    log.write("validation datasets loaded\n")
    # Get number of classes
    log.write("Get number of classes\n")
    num_classes = len(train_dataset.class_names)
    # Create and compile the model
    log.write("Create and compile the model\n")
    model = create_model((img_height, img_width, 3), num_classes)
    log.write("Model created\n")
    # Train the model
    epochs = 10
    log.write(f"Train the model with {epochs}\n")
    epochs = 10
    model.fit(
    train_dataset,
    epochs=epochs
    )
    log.write("Model trained\n")

    log.write("Save the model\n")
    model.save('/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/model/model.h5')
    log.write("Model saved\n")
    # Evaluate the model on the test dataset
    results = model.evaluate(validation_dataset)

    log.write(f"Test Loss = {results[0]}")
    log.write(f"Test Accuracy = {results[1]}")
    log.write(f"Test Precision = {results[2]}")
    log.write(f"Test Recall = {results[3]}")
    log.write(f"Test AUC = {results[4]}")

    # For detailed classification report (optional)
    y_true = []
    y_pred = []
    for img_batch, label_batch in validation_dataset:
        preds = model.predict(img_batch)
        y_true.extend(np.argmax(label_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=train_dataset.class_names))
except Exception as e:
    log.write("Error occured\n")
    log.write(str(e))
finally:
    log.close()