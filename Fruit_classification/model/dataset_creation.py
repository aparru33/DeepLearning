#%%
import os
from PIL import Image
import shutil
import random
import pandas as pd
import numpy as np
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

######################################################################################################
# %%

#%%
move_n_files(
    '/home/ubuntu/workspace/finovox_main/main-repo/backend/dataset/logo_classifier/other_copy/other/cropped_image',
    '/home/ubuntu/workspace/finovox_main/main-repo/backend/dataset/logo_classifier/train/others',
    100000)


move_n_files(
    '/home/ubuntu/workspace/finovox_main/main-repo/backend/dataset/logo_classifier/other_copy/other/cropped_image',
    '/home/ubuntu/workspace/finovox_main/main-repo/backend/dataset/logo_classifier/test/others',
    25000)


# %%
# find files that exist in both directories
import os

def common_files(dir1, dir2):
    # List all files in each directory
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))
    
    # Find common files
    common = files_dir1.intersection(files_dir2)
    
    return common

dir1 = '/home/ubuntu/workspace/finovox_main/main-repo/backend/dataset/logo_classifier/test/logos'
dir2 = '/home/ubuntu/workspace/finovox_main/main-repo/backend/dataset/logo_classifier/train/others'

common = common_files(dir1, dir2)
print(f"Common files between {dir1} and {dir2}:")
n=1
for filename in common:
    if '.' in filename:
        print(filename)
        # if n<9:
        #     os.remove(os.path.join(dir1, filename))
        # else:
        #     os.remove(os.path.join(dir2, filename))
        # if n==10:
        #     n=0
        # n+=1
#     #_path = os.path.join(dir1, filename)
    # if '.' in filename:
    #     os.remove(os.path.join(dir1, filename))
# #     os.remove(os.path.join(dir2, filename))

# %%
res = []
PATH = '/home/ubuntu/workspace/finovox_main/dl_project/DeepLearning/Fruit_classification/dataset'
for dir in os.listdir(PATH):
    print(f'----- {dir} -----')
    for d in os.listdir(f'{PATH}/{dir}'):
        for ddir in os.listdir(f'{PATH}/{d}/{dir}'):
            print(ddir)
            di = {'dataset' : ddir, 'category' :ddir }
            if di not in res:
                res.append({'dataset' : ddir, 'category' :ddir })
res= pd.DataFrame(res)
# %%
