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
#%%
# for d in os.listdir(IMAGES_DATASET):
#     if not os.path.exists(PROJECT_PATH+'dataset/train/'+d):
#         os.mkdir(PROJECT_PATH+'dataset/train/'+d)
#     move_files(IMAGES_DATASET+d, PROJECT_PATH+'dataset/train/'+d, 0.8)

# #%%
# for d in os.listdir(IMAGES_DATASET):
#     if not os.path.exists(PROJECT_PATH+'dataset/test/'+d):
#         os.mkdir(PROJECT_PATH+'dataset/test/'+d)
#     move_files(IMAGES_DATASET+d, PROJECT_PATH+'dataset/test/'+d, 1)

# #%%
# for item in os.listdir(IMAGES_DATASET):
#     item_path = os.path.join(IMAGES_DATASET, item)
#     if os.path.isdir(item_path):
#         new_name = item.split()[0]
#         new_path = os.path.join(IMAGES_DATASET, new_name)
#         os.rename(item_path, new_path)

# #%%
# cat = []
# for i in [FRUIT_360_DATASET+'train', FRUIT_360_DATASET+'train',
#           FRUITS_VEGETABLES_DATASET+'train', IMAGES_DATASET]:
#     for j in os.listdir(i):
#         if os.path.isdir(os.path.join(i,j)):
#             cat.append({"dir" :i,"cat": j})

# cat_by_folder = sorted(cat, key=lambda k: k['cat'])
# cats = sorted(list({i['cat'] for i in cat_by_folder}))


# #%%
# import os

# def rename_folders_Camel(_folder_path):
#     for item in os.listdir(_folder_path):
#         item_path = os.path.join(_folder_path, item)
#         if os.path.isdir(item_path):
#             new_name = item[0].upper() + item[1:]
#             new_path = os.path.join(_folder_path, new_name)
#             os.rename(item_path, new_path)

# #%%
# for f in os.listdir(FRUIT_360_DATASET + 'test'):
#     if os.path.isdir(FRUIT_360_DATASET + 'test/'+f):
#         for i in os.listdir(FRUIT_360_DATASET + 'test/'+f):
#             file_name = i.split('.')[0]
#             ext = i.split('.')[1]
#             os.rename(FRUIT_360_DATASET + 'test/'+f+'/'+i,
#                         FRUIT_360_DATASET + 'test/'+f+'/'+'360_'+ file_name +'.'+ext)
# %%
# # first word fruit having no multiple fruit in image
# to_remove = [
# 'Cactus',
#  'Cantaloupe',
#  'Carambula',
#  'Galia',
#  'Chestnut',
#  'Granadilla',
#  'Grapefruit',
#  'Guava',
#  'Huckleberry',
#  'Kaki',
#  'Kohlrabi',
#  'Kumquats',
#  'Limes',
#  'Lychee',
#  'Mangostan',
#  'Maracuja',
#  'Pepino',
#  'Pepper',
#  'Pepper',
#  'Pepper',
#  'Pepper',
#  'Physalis',
#  'Plum',
#  'Potato Red',
#  'Quince',
#  'Rambutan',
#  'Redcurrant',
#  'Salak',
#  'Tamarillo',
#  'Tangelo',
#  'Walnut',
#  'Nut',
#  'Cocos',
#  'Peas',

# ]
# for d in os.listdir(PROJECT_PATH + "dataset/train"):
#     if os.path.isdir(PROJECT_PATH + "dataset/train/"+d):
#         if d.split()[0] in to_remove:
#             shutil.rmtree(PROJECT_PATH + "dataset/train/"+d)


#%%
has_no_multiple= [
 'Blueberry',
 'Clementine',
 'Dates',
 'Fig',
 'Hazelnut',
 'Mandarine',
 'Peach',
 'Raspberry',]


has_no_multiple_2= [
 'Apricot',
 'Avocado',]


# Create an ImageDataGenerator with augmentation settings
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
    rotation_range=40,       # Randomly rotate images by up to 30 degrees
    zoom_range=[1.0, 1.2],   # Zoom-out only by up to 20% as the training image have max zoom in
    width_shift_range=0.02,  # Randomly shift images horizontally by up to 2%
                                #we don't want to alter the shape too much 
    height_shift_range=0.02, # Randomly shift images vertically by up to 2%
    fill_mode='nearest',     # Fill mode for handling empty pixels
    brightness_range=[0.8, 1.2],  # Randomly adjust brightness within a range
    rescale=1.0/255.0,        # Rescale pixel values to [0, 1] (if not already)
    horizontal_flip=True,    # Flip images horizontally
)

for s in ['train', 'test']:
    for d in os.listdir(PROJECT_PATH + "dataset/"+s):
        if os.path.isdir(PROJECT_PATH + "dataset/" +s +"/"+d):
            if d.split()[0] not in has_no_multiple +has_no_multiple_2:
                list_file = [PROJECT_PATH + "dataset/"+s+ "/"+d+"/"+f for f in os.listdir(PROJECT_PATH + "dataset/"+s+"/"+d)]
                nb_file = len(list_file)
                nb_to_generate = int((800-534)/nb_file) if s =='train' else int((200-134)/nb_file)
                for image_file in list_file:
                    img = load_img(image_file)
                    x = img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    # Generate augmented images and save them
                    i = 0
                    for batch in datagen.flow(x, batch_size=1):
                        augmented_image = array_to_img(batch[0])
                        file_ext = image_file.split(".")[-1]
                        augmented_image.save(os.path.join(PROJECT_PATH + "dataset/" +s+"/"+d, f'{image_file.split("/")[-1].split(".")[0]}_augmented_{i}.' + file_ext))
                        i += 1
                        if i >= nb_to_generate: 
                            break


# %%

for d in sorted(os.listdir(PROJECT_PATH + "dataset/test/")):
    if os.path.isdir(PROJECT_PATH + "dataset/test/"+d) and d.split()[0] in has_no_multiple:
        list_file = sorted([PROJECT_PATH + "dataset/test/"+d+"/"+f for f in os.listdir(PROJECT_PATH + "dataset/test/"+d)])
        to_kept = int(len(list_file)/200)
        for n in range(len(list_file)):
            if n%to_kept!=0:
                os.remove(list_file[n])
# %%
