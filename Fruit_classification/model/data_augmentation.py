import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Define the directory containing your signature dataset
#dataset_dir = 'train/set'

# the directories have the following structure:
# train
#    -fruit_specie_1
#    -fruit_specie_2
#    - ...
#    -fruit_specie_n

# test (contains only image, no directory)

directory = 'dataset/train/'
directories = [d.name for d in os.scandir(directory) if d.is_dir()]

# Create an ImageDataGenerator with augmentation settings
datagen = ImageDataGenerator(
    rotation_range=30,       # Randomly rotate images by up to 30 degrees
    zoom_range=[1.0, 1.2],   # Zoom-out only by up to 20% as the training image have max zoom in
    width_shift_range=0.02,  # Randomly shift images horizontally by up to 2%
                                #we don't want to alter the shape too much 
    height_shift_range=0.02, # Randomly shift images vertically by up to 2%
    fill_mode='nearest',     # Fill mode for handling empty pixels
    brightness_range=[0.8, 1.2],  # Randomly adjust brightness within a range
    rescale=1.0/255.0,        # Rescale pixel values to [0, 1] (if not already)
    horizontal_flip=True,    # Flip images horizontally
    seed=42                  # Use the same seed for consistency
)

for d in directories:
    # List all image files in the dataset directory
    output_dir = directory+d
    image_files = [os.path.join(output_dir, filename) for filename in os.listdir(output_dir) if (filename.endswith('.png') or filename.endswith('.jpg'))]
    
    # Make sure the output directory exists
    #os.makedirs(output_dir, exist_ok=True)

    # Loop through each image and apply data augmentation
    for image_file in image_files:
        # Load the image
        img = load_img(image_file)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Generate augmented images and save them
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            augmented_image = array_to_img(batch[0])
            file_ext = image_file.split(".")[-1]
            augmented_image.save(os.path.join(output_dir, f'{image_file.split("/")[-1].split(".")[0]}_augmented_{i}.' + file_ext))
            i += 1
            if i >= 10:  # Generate 10 augmented images for each original image
                break