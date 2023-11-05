import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import classification_report
import numpy as np

def create_model(input_shape, num_classes):
    model = models.Sequential()

    # Input Layer
    model.add(layers.Input(shape=input_shape))

    # Normalization Layer
    model.add(layers.Rescaling(1./255))

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Max Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Dropout Layer
    model.add(layers.Dropout(0.25))

    # Flatten Layer
    model.add(layers.Flatten())

    # Dense Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the Model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', Precision(), Recall(), AUC()])

    return model

# Set the path to your dataset
TRAIN_DIR = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/dataset/train'
VALIDATION_DIR = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/dataset/validation'

# Bilinear, bicubic, lanczos5, mitchellcubic.
INTERPOLATION = "bilinear"

# Load dataset
batch_size = 32
img_height = 64
img_width = 64

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
        label_mode='categorical'
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