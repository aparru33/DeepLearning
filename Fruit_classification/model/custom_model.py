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
TRAIN_DIR = 'D:/Workspace_Python/DeepLearning/Fruit_classification/dataset/train'
VALIDATION_DIR = 'D:/Workspace_Python/DeepLearning/Fruit_classification/dataset/validation'

# Bilinear, bicubic, lanczos5, mitchellcubic.
INTERPOLATION = "bilinear"

# Load dataset
batch_size = 32
img_height = 64
img_width = 64
train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)
validation_dataset = image_dataset_from_directory(
    VALIDATION_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

# Get number of classes
num_classes = len(train_dataset.class_names)
print(num_classes)
# Create and compile the model
model = create_model((img_height, img_width, 3), num_classes)

# Train the model
epochs = 10
model.fit(
  train_dataset,
  epochs=epochs
)


# Evaluate the model on the test dataset
results = model.evaluate(validation_dataset)
print("Test Loss:", results[0])
print("Test Accuracy:", results[1])
print("Test Precision:", results[2])
print("Test Recall:", results[3])
print("Test AUC:", results[4])

# For detailed classification report (optional)
y_true = []
y_pred = []
for img_batch, label_batch in validation_dataset:
    preds = model.predict(img_batch)
    y_true.extend(np.argmax(label_batch.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, target_names=train_dataset.class_names))