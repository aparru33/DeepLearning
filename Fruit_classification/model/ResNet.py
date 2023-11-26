# Import TensorFlow and Keras modules and its important APIs
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import io
import numpy as np
import os



# Setting Training Hyperparameters 
BATCH_SIZE = 256 # original ResNet paper uses batch_size = 128 for training 
EPOCHS = 300
#num_classes = 31

# Data Preprocessing 
subtract_pixel_mean = True
n = 2

# Select ResNet Version 
version = 1

# Computed depth of 
if version == 1: 
	depth = n * 6 + 2
elif version == 2: 
	depth = n * 9 + 2

# Model name, depth and version 
model_type = 'ResNet % dv % d' % (depth, version) 

# for test
# Load the CIFAR-10 data. 
#(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
PROJECT_PATH='/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/'
TRAIN_DIR = PROJECT_PATH+'dataset/final_dataset/train'
VALIDATION_DIR = PROJECT_PATH+'dataset/final_dataset/test'
INPUT_SHAPE = (224,224,3)
IMAGE_SIZE = (224,224)
log = open(PROJECT_PATH+f'ResNesT_log_{datetime.now()}.txt', "w")
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
# with cifar10
# # Input image dimensions. 
# input_shape = x_train.shape[1:] 

# # Normalize data. 
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# # If subtract pixel mean is enabled 
# if subtract_pixel_mean: 
# 	x_train_mean = np.mean(x_train, axis = 0) 
# 	x_train -= x_train_mean 
# 	x_test -= x_train_mean 

# # Print Training and Test Samples 
# print('x_train shape:', x_train.shape) 
# print(x_train.shape[0], 'train samples') 
# print(x_test.shape[0], 'test samples') 
# print('y_train shape:', y_train.shape) 

# Convert class vectors to binary class matrices. 
# y_train = tf.keras.utils.to_categorical(y_train, num_classes) 
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Setting LR for different number of Epochs 
def lr_schedule(epoch): 
    lr = 1e-3  # Default learning rate
    if epoch > 350: 
        lr *= 0.8e-4
    elif epoch > 300: 
        lr *= 0.1e-3
    elif epoch > 250: 
        lr *= 0.5e-3
    elif epoch > 200: 
        lr *= 1e-3
    elif epoch > 150: 
        lr *= 1e-2
    elif epoch > 100: 
        lr *= 1e-1

    print('Learning rate:', lr) 
    return lr

# Basic ResNet Building Block 


def resnet_layer(inputs, 
				num_filters=16, 
				kernel_size=3, 
				strides=1, 
				activation='relu', 
				batch_normalization=True,
				conv_first=True):
	
	conv=Conv2D(num_filters, 
				kernel_size=kernel_size, 
				strides=strides, 
				padding='same', 
				kernel_initializer='he_normal', 
				kernel_regularizer=l2(1e-4))

	x=inputs 
	if conv_first: 
		x = conv(x) 
		if batch_normalization: 
			x = BatchNormalization()(x) 
		if activation is not None: 
			x = Activation(activation)(x) 
	else: 
		if batch_normalization: 
			x = BatchNormalization()(x) 
		if activation is not None: 
			x = Activation(activation)(x) 
		x = conv(x) 
	return x 

def resnet_v1(input_shape, depth, num_classes=31): 

	if (depth - 2) % 6 != 0: 
		raise ValueError('depth should be 6n + 2 (eg 20, 32, 44 in [a])') 
	# Start model definition. 
	num_filters = 16
	num_res_blocks = int((depth - 2) / 6) 

	inputs = Input(shape=input_shape) 
	x = resnet_layer(inputs=inputs) 
	# Instantiate the stack of residual units 
	for stack in range(3): 
		for res_block in range(num_res_blocks): 
			strides = 1
			if stack > 0 and res_block == 0: # first layer but not first stack 
				strides = 2 # downsample 
			y = resnet_layer(inputs=x, 
							num_filters=num_filters, 
							strides=strides) 
			y = resnet_layer(inputs=y, 
							num_filters=num_filters, 
							activation=None) 
			if stack >0 and res_block == 0: # first layer but not first stack 
				# linear projection residual shortcut connection to match 
				# changed dims 
				x = resnet_layer(inputs=x, 
								num_filters=num_filters, 
								kernel_size=1, 
								strides=strides, 
								activation=None, 
								batch_normalization=False) 
			x = layers.add([x, y]) 
			x = Activation('relu')(x) 
		num_filters *= 2

	# Add classifier on top. 
	# v1 does not use BN after last shortcut connection-ReLU 
	x = AveragePooling2D(pool_size=8)(x) 
	y = Flatten()(x) 
	outputs = Dense(num_classes, 
					activation='softmax', 
					kernel_initializer='he_normal')(y) 

	# Instantiate model. 
	model = Model(inputs=inputs, outputs=outputs) 
	return model 

# ResNet V2 architecture 
def resnet_v2(input_shape, depth, num_classes=31): 
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n + 2 (eg 56 or 110 in [b])')
    
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2  # downsample

            y = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1, strides=strides, activation=activation, batch_normalization=batch_normalization, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False)
            
            if res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters_out, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
            
            x = layers.add([x, y])

        num_filters_in = num_filters_out

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model


try:
    start_time=datetime.now()
    # Main function 
    if version == 2: 
        model = resnet_v2(input_shape = INPUT_SHAPE, depth = depth) 
    else: 
        model = resnet_v1(input_shape = INPUT_SHAPE, depth = depth) 

    model.compile(loss ='categorical_crossentropy', 
                optimizer = Adam(learning_rate = lr_schedule(0)), 
                metrics =['accuracy'])

	# Capture the model summary
    summary_str = io.StringIO()
    model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
    model_summary = summary_str.getvalue()
    summary_str.close()

    # Log the model summary
    log.write(model_summary +'\n')
    log.write(model_type+"\n")

    # Prepare model saving directory. 
    save_dir = PROJECT_PATH +'saved_model'
    model_name = 'resnet% s_model.{epoch:03d}.h5' % model_type 
    if not os.path.isdir(save_dir): 
        os.makedirs(save_dir) 
    filepath = os.path.join(save_dir, model_name) 

    # Prepare callbacks for model saving and for learning rate adjustment. 
    checkpoint = ModelCheckpoint(filepath = filepath, 
                                monitor ='val_accuracy', 
                                verbose = 1, 
                                save_best_only = True) 

    lr_scheduler = LearningRateScheduler(lr_schedule) 

    lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1), 
                                cooldown = 0, 
                                patience = 5, 
                                min_lr = 0.5e-6, 
                                monitor='val_loss'  ) 
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, 
                                patience=10, verbose=1, mode='min', 
                                restore_best_weights=True)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping] 

    # Run training
    log.write(f"Train the models with {EPOCHS} epochs\n")
    model.fit(train_dataset, 
            epochs = EPOCHS, 
            validation_data=validation_dataset,
            callbacks = callbacks) 
    log.write("Model trained\n")
    # Evaluate the model on the test dataset
    results = model.evaluate(validation_dataset)
    log.write("Model evaluated\n")

    log.write(", ".join([str(r)+"\n" for r in results]) + "\n")

    log.write("Duration of training = " + str(datetime.now() - start_time) + "\n")

	
except Exception as e:
    log.write("Error\n")
    log.write(str(e))
    log.write("Duration of training = " + str(datetime.now() - start_time) + "\n")
finally:   
    log.close()