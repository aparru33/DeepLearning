#%%
import os
from PIL import Image
import shutil
import random
import pandas as pd
import os

#%%
def move_files(_source_directory, _target_directory, percentage=0.47):
    _files = [f for f in os.listdir(_source_directory) if os.path.isfile(os.path.join(_source_directory, f))]

    # Shuffle the files to ensure randomness
    if percentage < 1.0:
        random.shuffle(_files)
        # Calculate the number of files to move
        num_files_to_move = int(percentage * len(_files))
    else:
        num_files_to_move = len(_files)
    # Move the files
    for i in range(num_files_to_move):
    #for i in range(len(all_files)):
        source_file_path = os.path.join(_source_directory, _files[i])
        target_file_path = os.path.join(_target_directory, _files[i])
        shutil.move(source_file_path, target_file_path)


def move_n_files(_source_directory, _target_directory,n):
    _files = [f for f in os.listdir(_source_directory) if os.path.isfile(os.path.join(_source_directory, f))]

    # Shuffle the files to ensure randomness
    if n > len(_files):
        print("n is greater than number of files")
        return
    # Move the files

    random.shuffle(_files)
    for i in range(n):
    #for i in range(len(all_files)):
        source_file_path = os.path.join(_source_directory, _files[i])
        target_file_path = os.path.join(_target_directory, _files[i])
        shutil.move(source_file_path, target_file_path)

def move_files_subdir(_source_directory, _target_directory, percentage=0.47):
    # Ensure the target directory exists
    if not os.path.exists(_target_directory):
        os.makedirs(_target_directory)

    # Iterate over all subdirectories in the source directory
    for d in os.listdir(_source_directory):
        print(d)
        subdir_path = os.path.join(_source_directory, d)
        
        # Check if it's indeed a directory
        if os.path.isdir(subdir_path):
           move_files(subdir_path, _target_directory, percentage)

def copy_files(source_directory, target_directory, percentage=0.47):
    all_files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]
    
    # Shuffle the files to ensure randomness
    if percentage < 1.0:
        random.shuffle(all_files)
        
        # Calculate the number of files to move
        num_files_to_move = int(percentage * len(all_files))
    else:
        num_files_to_move = len(all_files)
    # Move the files
    for i in range(num_files_to_move):
    #for i in range(len(all_files)):
        source_file_path = os.path.join(source_directory, all_files[i])
        target_file_path = os.path.join(target_directory, all_files[i])
        
        # Ensure there's no filename collision. If so, you might need a more complex renaming scheme.
        #if not os.path.exists(target_file_path):
        #    shutil.move(source_file_path, target_file_path)
        #else:
        # A simple renaming scheme: append the subdir name to the filename.
        #new_target_file_path = os.path.join(target_directory, f"{subdir_name}_{all_files[i]}")
        shutil.copy2(source_file_path, target_file_path)



def copy_files_subdir(source_directory, target_directory, percentage=0.47):
    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Iterate over all subdirectories in the source directory
    for subdir_name in os.listdir(source_directory):
        subdir_path = os.path.join(source_directory, subdir_name)
        
        # Check if it's indeed a directory
        if os.path.isdir(subdir_path):
           copy_files(subdir_path, target_directory, percentage)


def compute_area(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])


def copy_file(file_name):
    """Function to copy a file from source to destination directory"""
    source_file_path = os.path.join(source_dir, file_name)
    dest_file_path = os.path.join(dest_dir, file_name)

    try:
        shutil.copy2(source_file_path, dest_file_path)
    except Exception as e:
        print(f"Failed to copy {file_name}: {e}")


def get_all_files_in_directory(dir):
    """Recursively get all files in the directory and its subdirectories"""
    for root, _, files in os.walk(dir):
        for file in files:
            yield os.path.join(root, file)

def common_files(dir1, dir2):
    # List all files in each directory
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))
    
    # Find common files
    common = files_dir1.intersection(files_dir2)
    
    return common
######################################################################################################
# %%
# first we get all the catgories in each directory
PROJECT_PATH = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/'
ZHANG_DATASET = PROJECT_PATH + "dataset/fruit_classification_zhang/"
FRUIT_360_DATASET = PROJECT_PATH + "dataset/Fruits_360/"
FRUITS_VEGETABLES_DATASET = PROJECT_PATH + "dataset/Fruits_and_Vegetables_Image_Recognition_Dataset/"
IMAGES_DATASET = PROJECT_PATH + "dataset/images/"
FINAL_DATASET = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/dataset_original_size_before_augment/'
FINAL_DATASET_224 = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/dataset/final_dataset/'
#%%
for d in os.listdir(FINAL_DATASET+'train'):
    if not os.path.exists(FINAL_DATASET+'test/'+d):
        os.mkdir(FINAL_DATASET+'test/'+d)
    move_files(FINAL_DATASET+'train/'+d, FINAL_DATASET+'test/'+d, 0.2)

# #%%
# for d in os.listdir(IMAGES_DATASET):
#     if not os.path.exists(PROJECT_PATH+'dataset/test/'+d):
#         os.mkdir(PROJECT_PATH+'dataset/test/'+d)
#     move_files(IMAGES_DATASET+d, PROJECT_PATH+'dataset/test/'+d, 1)

#%%
for item in os.listdir(IMAGES_DATASET+'/images'):
    item_path = os.path.join(IMAGES_DATASET+'/images', item)
    if os.path.isdir(item_path):
        new_name = item.split()[0]
        new_name = new_name[0].lower()+new_name[1:]
        new_path = os.path.join(IMAGES_DATASET+'/images', new_name)
        # print(new_path)
        # if new_name in os.listdir(FINAL_DATASET+'/train/'):
        #     for f in os.listdir(item_path):
        #         shutil.move(FINAL_DATASET+'/train/'+item+'/' + f, 
        #                     FINAL_DATASET+'/train/'+new_name+'/' + f)
        #     shutil.rmtree(item_path)
        # else:
        os.rename(item_path, new_path)

# #%%
# cat = []
# for i in [FRUIT_360_DATASET+'train', FRUIT_360_DATASET+'train',
#           FRUITS_VEGETABLES_DATASET+'train', IMAGES_DATASET]:
#     for j in os.listdir(i):
#         if os.path.isdir(os.path.join(i,j)):
#             cat.append({"dir" :i,"cat": j})

# cat_by_folder = sorted(cat, key=lambda k: k['cat'])
# cats = sorted(list({i['cat'] for i in cat_by_folder}))

#%%
# Create an ImageDataGenerator with augmentation settings
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    width_shift_range=0.02,
    height_shift_range=0.02,  # Shift the images vertically by up to 10% of their height
    shear_range=0.01,         # Shear the images by up to 10 degrees
    zoom_range=0.1,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    rescale=1.0/255.0,
    horizontal_flip=True,
    vertical_flip=True,
    )
import math
def augment_images(directory, max_files, datagen):
    current_file_count = len([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
    nb_to_create = max_files - current_file_count
    augmentations_per_image = math.ceil(nb_to_create / current_file_count)
    generated_images = 0
    for image_file in os.listdir(directory):
        if generated_images >= nb_to_create:
            break
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(directory, image_file)
            image = load_img(image_path)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # Augment the image
            p = 0
            for batch in datagen.flow(image, batch_size=1, save_to_dir=directory, save_prefix='aug_', save_format='jpg'):
                generated_images += 1
                p+=1
                if generated_images >= max_files or p>augmentations_per_image:
                    break
    print(f"Generated {generated_images} images for {directory} \n")         
# for d in os.listdir(FINAL_DATASET_224+'test'):
#     augment_images(FINAL_DATASET_224+'test/'+d, 200, datagen)


import multiprocessing

def process_directory(directory):
    # Your method logic here
    print(f"Processing {directory}")

def start_process_for_directory(directory, max_files):
    process = multiprocessing.Process(target=augment_images, args=(directory,max_files,datagen))
    process.start()
    return process

directories = os.listdir(FINAL_DATASET_224+'train/')  # List of directories

# Start a process for each directory
processes = [start_process_for_directory(FINAL_DATASET_224+'train/'+dir, 800) for dir in directories]

# Wait for all processes to finish
for process in processes:
    process.join()

print("All directories processed for test.")

# %%

for d in sorted(os.listdir(PROJECT_PATH + "dataset/test/")):
    if os.path.isdir(PROJECT_PATH + "dataset/test/"+d) and d.split()[0] :
        list_file = sorted([PROJECT_PATH + "dataset/test/"+d+"/"+f for f in os.listdir(PROJECT_PATH + "dataset/test/"+d)])
        to_kept = int(len(list_file)/200)
        for n in range(len(list_file)):
            if n%to_kept!=0:
                os.remove(list_file[n])
# %%

import os
import random

def reduce_files_to_limit(directory, file_limit=800):
    """
    Randomly delete files in the specified directory until the number of files is equal to file_limit.
    Args:
    - directory (str): The path to the directory.
    - file_limit (int): The maximum number of files to retain in the directory.
    """
    # Get a list of files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Check if the number of files exceeds the limit
    while len(files) > file_limit:
        # Randomly select a file to delete
        file_to_delete = random.choice(files)
        
        # Delete the file
        os.remove(os.path.join(directory, file_to_delete))
        
        # Update the files list
        files.remove(file_to_delete)

# Example usage
directory_path = '/path/to/your/directory'
for d in os.listdir(PROJECT_PATH + "dataset/train/"):
    if os.path.isdir(PROJECT_PATH + "dataset/train/"+d):
        reduce_files_to_limit(PROJECT_PATH + "dataset/train/"+d, 800)


for d in os.listdir(PROJECT_PATH + "dataset/test/"):
    if os.path.isdir(PROJECT_PATH + "dataset/test/"+d):
        reduce_files_to_limit(PROJECT_PATH + "dataset/test/"+d, 200)

# %%
import os
from PIL import Image

def find_min_image_shape(root_dir):
    min_shape = None
    dc ={}
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            try:
                filepath = os.path.join(subdir, file)
                with Image.open(filepath) as img:
                    dc[img.size] = dc.get(img.size, 0) + 1
                    if min_shape is None:
                        min_shape = img.size  # img.size is in (width, height) format
                    else:
                        # Compare current image's size to the minimum found so far
                        min_shape = tuple(min(s, m) for s, m in zip(img.size, min_shape))
            except IOError:
                # If the file cannot be opened as an image, skip it
                print(f"Skipping non-image file: {filepath}")
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")

    # Return in (height, width) format
    return [min_shape[1], min_shape[0] if min_shape else [None,None]], dc

# Example usage:
root_dir = FINAL_DATASET
min_shape, dc = find_min_image_shape(root_dir)
print(f"The minimum image shape in the directory and subdirectories is: {min_shape}")
for k,v in {key: dc[key] for key in sorted(dc)}.items():
    print(k,v)

# %%
import os

def find_min_files_directory(root_dir):
    min_files = float('inf')
    min_dir = None
    res = []
    for subdir, dirs, files in os.walk(root_dir):
        if dirs:
            continue
        file_count = sum([os.path.isfile(os.path.join(subdir, f)) for f in files])
        res.append((subdir, file_count))

    return  res

# Example usage
root_dir = FINAL_DATASET_224+'/train'
ll = find_min_files_directory(root_dir)
for tu in ll:   
    directory_with_min_files,min_files_count = tu  
    print(f"Directory with the minimum number of files: {directory_with_min_files.split('/')[-1]}")
    print(f"Number of files in this directory: {min_files_count} \n")

# %%
import os
from PIL import Image

def remove_min_image_shape(root_dir):
    min_shape = None
    dc ={}
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            try:
                filepath = os.path.join(subdir, file)
                if filepath.endswith(".png") or filepath.endswith(".jpg"):
                    # Open the image and check its shape
                    with Image.open(filepath) as img:
                        size = img.size
                        height, width = size[0], size[1]
                    if height<224 or width<224:
                        os.remove(filepath)
                elif height>width*2 or width>height*2:
                    os.remove(filepath)
                
                        
               
            except IOError:
                # If the file cannot be opened as an image, skip it
                print(f"Skipping non-image file: {filepath}")
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")

remove_min_image_shape('/Users/tech/Downloads/archive/Fruit-262')


#%%
from PIL import Image

def resize_and_pad(path, dir, cat,f, target_size=(224, 224), fill_color=(0, 0, 0)):

    img_path = path+dir+'/'+cat+'/'+f
    file_name = f.split('/')[-1]
    name = file_name.split('.')[0]
    ext = file_name.split('.')[1]
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
    new_img.save(FINAL_DATASET_224+dir+'/'+cat+'/'+name+'_224.'+ext)

for ds in ['train','test']:
    print('\n ##############')
    print(ds)
    for d in os.listdir(FINAL_DATASET+ds):
        print(d)
        for f in os.listdir(FINAL_DATASET+ds+'/'+d):
            resize_and_pad(FINAL_DATASET,ds,d,f)

# %%
