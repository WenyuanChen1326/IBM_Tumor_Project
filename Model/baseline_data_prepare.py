import csv
import os
from collections import defaultdict
import numpy as np
import contextlib
import logging
from skimage.feature import peak_local_max
import pickle
import h5py
from tqdm import tqdm
import nibabel as nib


# Function to read the metadata CSV and organize it
# def read_metadata(metadata_path):
#     patients_data = defaultdict(dict)
#     with open(metadata_path, newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             patient_id = row['study_location'].split('/')[3]  # Extracting the patient_id from the path
#             if patient_id not in patients_data:
#                 patients_data[patient_id] = {
#                     'studies': [],
#                     'diagnosis': row['diagnosis'],
#                     'age': row['age'],
#                     'sex': row['sex']
#                 }
#             patients_data[patient_id]['studies'].append(row['study_location'])
#     return patients_data
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def check_coord_boundaries(coord, block_size, shape):
    # Calculate the center coordinates
    x_center, y_center, z_center = coord

    # Calculate half sizes for each dimension, adjusting for even sizes
    half_size_x, half_size_y, half_size_z = [size // 2 for size in block_size]

    # Calculate start and end, ensuring they are within the array bounds
    x_start = max(x_center - half_size_x, 0)
    x_end = min(x_start + block_size[0], shape[0])
    y_start = max(y_center - half_size_y, 0)
    y_end = min(y_start + block_size[1], shape[1])
    z_start = max(z_center - half_size_z, 0)
    z_end = min(z_start + block_size[2], shape[2])

    # Correct the start positions if the block exceeds the mask dimensions
    if x_end - x_start < block_size[0]: x_start = x_end - block_size[0]
    if y_end - y_start < block_size[1]: y_start = y_end - block_size[1]
    if z_end - z_start < block_size[2]: z_start = z_end - block_size[2]
    return x_start, y_start, z_start

def save_blocks_hdf5(output_folder,study_id, positive_blocks, negative_blocks):
    # patient_path = os.path.join(output_folder, study_id)
    # # patient_path = os.path.join(output_folder, patient_path)
    # print(patient_path)
    # os.makedirs(patient_path, exist_ok=True)
    
    # hdf5_filename = f'{patient_path}_blocks.hdf5'
    # # hdf5_filepath = os.path.join(patient_path, hdf5_filename)
    # hdf5_filepath = os.path.join(patient_path, hdf5_filename)
        # Correct the directory path for saving the HDF5 file
    hdf5_directory = os.path.join(output_folder, study_id)
    os.makedirs(hdf5_directory, exist_ok=True)
    
    hdf5_filename = f'{study_id}_blocks.hdf5'  # HDF5 file should only be named after the study
    hdf5_filepath = os.path.join(hdf5_directory, hdf5_filename)
    with h5py.File(hdf5_filepath, 'w') as hdf5_file:
        if positive_blocks:
            hdf5_file.create_dataset('positive', data=np.stack(positive_blocks), compression='gzip')
        else:
            # Create an empty dataset or handle accordingly if no positive blocks are present
            hdf5_file.create_dataset('positive', data=np.array([]), compression='gzip')

        # hdf5_file.create_dataset('positive', data=np.stack(positive_blocks), compression='gzip')
        hdf5_file.create_dataset('negative', data=np.stack(negative_blocks), compression='gzip')

def process_patient_folder(patient_id, study_id, block_size = (3, 3, 3), min_distance = 1, threshold_abs = 2.0, 
                           data_root = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data', 
                           output_folder = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Processed_Block'):
    patient_folder = os.path.join(data_root, patient_id)
    subfolder_path = os.path.join(patient_folder, study_id) 
    seg_file_path = os.path.join(subfolder_path, 'SEG.nii.gz')
    suv_file_path = os.path.join(subfolder_path, 'SUV.nii.gz')

    seg_img = nib.load(seg_file_path)
    seg_data = seg_img.get_fdata()
    suv_img = nib.load(suv_file_path)
    suv_data = suv_img.get_fdata()

    voxel_dimensions = suv_img.header['pixdim'][1:4]
    assert voxel_dimensions[0] == 2.0364201068878174, f"voxel_dimensions[0] isn't matched: {voxel_dimensions[0]}"
    assert voxel_dimensions[1] ==  2.0364201068878174, f"voxel_dimensions[1] isn't matched: {voxel_dimensions[1]}"
    assert voxel_dimensions[2] == 3, f"voxel_dimensions[2] isn't matched: {voxel_dimensions[2]}"

    R=min_distance # change it to 1 ?
    TH=threshold_abs # change it to smaller ones
    local_max=peak_local_max(suv_data,min_distance=R,threshold_abs=TH)
    positive_blocks = []
    negative_blocks = []
    for coord in local_max:
        x_start, y_start, z_start = check_coord_boundaries(coord, block_size, seg_data.shape)
        suv_block = suv_data[x_start:x_start+block_size[0], y_start:y_start+block_size[1], z_start:z_start+block_size[2]]
        seg_block = seg_data[x_start:x_start+block_size[0], y_start:y_start+block_size[1], z_start:z_start+block_size[2]]
        if np.any(seg_block):
            positive_blocks.append(suv_block)
        else:
            negative_blocks.append(suv_block)
    output_folder += f"_block_size{block_size[0]}"
    output_folder = os.path.join(output_folder, patient_id)
    save_blocks_hdf5(output_folder, study_id, positive_blocks, negative_blocks)
    return len(positive_blocks), len(negative_blocks)

def main(data_directory, block_size = (3, 3, 3), min_distance = 1, threshold_abs = 2.0):
    patients = sorted([pid for pid in os.listdir(data_directory) if not pid.startswith('.') and os.path.isdir(os.path.join(data_directory, pid))])
    for patient_id in tqdm(patients, desc="Processing patients"):
        if patient_id == 'PETCT_1285b86bea':
            continue
        patient_folder = os.path.join(data_directory, patient_id)
        subfolders = sorted([f.name for f in os.scandir(patient_folder) if f.is_dir() and not f.name.startswith('.')])
        for study_id in subfolders:
            pos_count, neg_count = process_patient_folder(patient_id, study_id, block_size = block_size, min_distance = min_distance, threshold_abs = threshold_abs, data_root = data_directory)
            pos_block_count[block_size] += pos_count
            neg_block_count[block_size] += neg_count
            # process_patient_folder(patient_id, study_id, block_size = block_size, min_distance = min_distance, threshold_abs = threshold_abs, data_root = data_directory)
            # break
        # break
if __name__ == '__main__':
    block_size_3 = (3, 3, 3)
    block_size_16 = (16, 16, 16)
    # Initialize counters
    pos_block_count = {block_size_3: 0, block_size_16: 0}
    neg_block_count = {block_size_3: 0, block_size_16: 0}
    # data_directory = "/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data"
    data_directory = "/Volumes/T7 Shield/FDG-PET-CT-Lesions"
    print(f"start {block_size_3}!")
    main(data_directory, block_size = block_size_3)
    
    print(f"end {block_size_3}!")

    print(f"start {block_size_16}!")
    main(data_directory, block_size = block_size_16)
    print(f"end {block_size_16}!")
    with open('block_counts.txt', 'w') as file:
        file.write('Block Size 3x3x3:\n')
        file.write(f'Positive Blocks: {pos_block_count[block_size_3]}\n')
        file.write(f'Negative Blocks: {neg_block_count[block_size_3]}\n')
        file.write('\n')
        file.write('Block Size 16x16x16:\n')
        file.write(f'Positive Blocks: {pos_block_count[block_size_16]}\n')
        file.write(f'Negative Blocks: {neg_block_count[block_size_16]}\n')

    '''def load_hdf5_data(file_path, dataset_name):
        """
        Load a dataset from an HDF5 file.

        Parameters:
        - file_path: The path to the HDF5 file.
        - dataset_name: The name of the dataset within the HDF5 file to load (e.g., 'positive' or 'negative').

        Returns:
        - A NumPy array containing the data from the specified dataset.
        """
        with h5py.File(file_path, 'r') as hdf5_file:
            if dataset_name in hdf5_file:
                return np.array(hdf5_file[dataset_name])
            else:
                raise KeyError(f"Dataset {dataset_name} not found in file {file_path}")

    # Example usage:
    # Replace 'your_patient_id_blocks.hdf5' with the path to your actual HDF5 file.
    # Replace 'positive' with 'negative' to load negative blocks instead.
    file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Processed_Block/PETCT_f21755a99b/05-05-2005-NA-PET-CT Ganzkoerper  primaer mit KM-44651/05-05-2005-NA-PET-CT Ganzkoerper  primaer mit KM-44651_blocks.hdf5'
    # file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Processed_Block/PETCT_fe705ea1cc/04-27-2003-NA-Unspecified CT ABDOMEN-47025/04-27-2003-NA-Unspecified CT ABDOMEN-47025_blocks.hdf5'
    try:
        positive_blocks = load_hdf5_data(file_path, 'negative')
        # print(len(positive_blocks))
        print(positive_blocks.shape)
        # Now you can use the loaded positive_blocks for your analysis or model training.
    except KeyError as e:
        print(e)
'''

# Assuming your data folder and the metadata CSV file are in the same directory as your script
# metadata_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/autoPETmeta.csv'  # Replace with the actual path to your CSV file
# patients_data = read_metadata(metadata_path)
# print(patients_data)

# This will give you a dictionary with patient_ids as keys and their metadata + study paths as values
