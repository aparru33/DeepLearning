import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.layers import Dropout

from datetime import datetime
import io
import numpy as np
import os

BATCH_SIZE = 256  
EPOCHS = 400

PROJECT_PATH='/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/'
TRAIN_DIR = PROJECT_PATH+'dataset/final_dataset/train'
VALIDATION_DIR = PROJECT_PATH+'dataset/final_dataset/test'
INPUT_SHAPE = (224,224,3)
IMAGE_SIZE = (224,224)

try:
    start_time=datetime.now()
    log = open(PROJECT_PATH+f'efficientNetV2S_log_{datetime.now()}.txt', "w")
    log.write("Load datasets\n")
    log.write("Load datasets\n")
    train_dataset = image_dataset_from_directory(
            TRAIN_DIR,
            seed=123,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=True,
        )
    log.write("train datasets loaded\n")
    validation_dataset = image_dataset_from_directory(
    VALIDATION_DIR,
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    )
    num_classes = len(train_dataset.class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    log.write(" Load EfficientNetV2S\n")
    base_model = EfficientNetV2S(weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x) 
    x = Dropout(0.5)(x) 
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='efficientNetV2S_model_checkpoint.h5', save_best_only=True)

    model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, callbacks=[lr_schedule, early_stopping, model_checkpoint], verbose=1)

    total_layers = len(base_model.layers)
    N = int(total_layers * 0.2)  # Unfreezing last 20% of the layers
    for layer in base_model.layers[-N:]:  # Unfreeze the last N layers
        layer.trainable = True

    # Compile with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Capture the model summary
    summary_str = io.StringIO()
    model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
    model_summary = summary_str.getvalue()
    summary_str.close()
    model.save(PROJECT_PATH +f'efficientNetV2S_model_{str(start_time)}.keras')
    
    log.write("Model saved\n")
    # Evaluate the model on the test dataset
    results = model.evaluate(validation_dataset)
    log.write("Model evaluated\n")

    log.write(", ".join([str(r)+"\n" for r in results]) + "\n")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=PROJECT_PATH + 'efficientNetV2S_best_model.h5', save_best_only=True, monitor='val_accuracy')

    log.write("Duration of training = " + str(datetime.now() - start_time) + "\n")
except Exception as e:
    log.write("Error: " + str(e) + "\n")
finally:
    log.close() 