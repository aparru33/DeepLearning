#%%
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#%%
# Define the directory containing your signature dataset
#dataset_dir = 'train/set'

# the directories have the following structure:
# train
#    -fruit_specie_1
#    -fruit_specie_2
#    - ...
#    -fruit_specie_n

# test (contains only image, no directory)

directory = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/dataset/images/'
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

#%%
import os
import pandas as pd
import shutil

# Paths
csv_file_path = 'D:/Workspace_Python/DeepLearning/Fruit_classification/dataset/sampleSubmission.csv'
images_directory_path = 'D:/Workspace_Python/DeepLearning/Fruit_classification/dataset/test'
output_directory_path = 'D:/Workspace_Python/DeepLearning/Fruit_classification/dataset/sorted_test'

# Read CSV file
df = pd.read_csv(csv_file_path)

# Ensure 'id' is in string format and properly formatted
df['id'] = df['id'].apply(lambda x: str(x).zfill(4))

# Create directories for each label
for label in df['label'].unique():
    label_dir = os.path.join(output_directory_path, str(label))
    if not os.path.exists(label_dir):
        print("create", label_dir)
        os.makedirs(label_dir)

#%%
import os
import shutil
from math import ceil
import random

def move_files(src_dir, dst_dir, percentage=0.2):
    if not os.path.exists(src_dir):
        print("Source directory does not exist!")
        return
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Loop through each subfolder in the source directory
    for subfolder in os.listdir(src_dir):
        subfolder_path = os.path.join(src_dir, subfolder)

        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)
            files = [f for f in files if os.path.isfile(os.path.join(subfolder_path, f))]  # Filter out subdirectories
            num_files_to_move = ceil(percentage * len(files))

            # Randomly select files to move
            files_to_move = random.sample(files, num_files_to_move)

            # Create corresponding subfolder in destination directory if it doesn't exist
            dst_subfolder_path = os.path.join(dst_dir, subfolder)
            if not os.path.exists(dst_subfolder_path):
                os.makedirs(dst_subfolder_path)

            # Move files
            for file in files_to_move:
                src_file_path = os.path.join(subfolder_path, file)
                dst_file_path = os.path.join(dst_subfolder_path, file)
                shutil.move(src_file_path, dst_file_path)

            print(f"Moved {num_files_to_move} files from {subfolder_path} to {dst_subfolder_path}")

print("Moving files...")
move_files(
    'D:/Workspace_Python/DeepLearning/Fruit_classification/dataset/train', 
    'D:/Workspace_Python/DeepLearning/Fruit_classification/dataset/validation')
print("Files moved successfully!")


# %%
