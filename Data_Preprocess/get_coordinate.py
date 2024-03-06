from utils import get_connected_components_3D
import numpy as np
from utils import *
from tqdm import tqdm
from create_pos_block import *
from random_sampling_neg_block import *
import csv
import os
import contextlib
import logging
from skimage.feature import peak_local_max
import pickle


logging.basicConfig(
    filename='get_ct_coorinate_only.log', 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger()
def create_negative_blocks(seg_data, local_max,sample_size, block_size):
    negative_coordinates = sample_neg_block(seg_data,local_max,block_size, sample_size)
    return negative_coordinates
def create_non_restricted_pos_block(seg_data,positve_coordinates,sample_size, block_size):
    positve_coordinates =  sample_neg_block(seg_data,positve_coordinates,block_size, sample_size, negative=False)
    return positve_coordinates

def create_positive_blocks(filtered_separate_segmentation_masks):
    positive_coordinates = get_tumor_under_threshold_blocks_per_study(filtered_separate_segmentation_masks, block_size = (3, 3, 3))
    return positive_coordinates

def process_patient_folder(subfolder_path, sample_size = 100, block_size = (3, 3, 3)):
    seg_file_path = os.path.join(subfolder_path, 'SEG.nii.gz')
    suv_file_path = os.path.join(subfolder_path, 'SUV.nii.gz')
    ct_file_path = os.path.join(subfolder_path, 'CTres.nii.gz')


    seg_img = nib.load(seg_file_path)
    seg_data = seg_img.get_fdata()
    voxel_dimensions = seg_img.header['pixdim'][1:4]

    suv_img = nib.load(suv_file_path)
    suv_data = suv_img.get_fdata()

    ct_img = nib.load(ct_file_path)
    ct_data = ct_img.get_fdata()

    separate_segmentation_masks = get_connected_components_3D(seg_data)
    filtered_separate_segmentation_masks = filter_separate_segmentation_mask_by_diameter_and_SUV_max_and_voxel_of_interest(
        suv_data, voxel_dimensions, separate_segmentation_masks, 
        diameter_in_cm = 0.6, SUV_max = 3, voxel_of_interst = 3)
    R=3
    TH=2.0
    local_max=peak_local_max(suv_data,min_distance=R,threshold_abs=TH)
    positive_coordinates = create_positive_blocks(filtered_separate_segmentation_masks)
    negative_coordinates = create_negative_blocks(seg_data,local_max,sample_size, block_size)
    # Find the indices where the value is 1
    coordinates = np.where(seg_data == 1)
    # The returned coordinates are in the form of (x_indices, y_indices, z_indices)
    # To get a list of (x, y, z) tuples:
    pos_xyz_coordinates = np.array(list(zip(coordinates[0], coordinates[1], coordinates[2])))

    non_restricted_positive_coordinates = create_non_restricted_pos_block(seg_data,pos_xyz_coordinates,sample_size, block_size)
    return {
        'ct_data': ct_data,
        'suv_data': suv_data,
        'positive_coordinates': positive_coordinates,
        'negative_coordinates': negative_coordinates,
        'non_restricted_positive_coordinates': non_restricted_positive_coordinates
    }

def get_coorindates(data_directory, modality = 'pet', block_size = (3, 3, 3)):
    assert modality in ['pet', 'ct'], "modality must be either 'pet', 'ct'"
    patients = sorted([pid for pid in os.listdir(data_directory) if not pid.startswith('.') and os.path.isdir(os.path.join(data_directory, pid))])
    raw_input_data = []  # List to collect data for raw_input_df
    # raw_input_label = []
  
    for patient_id in tqdm(patients, desc="Processing patients"):
            if patient_id == 'PETCT_1285b86bea':
                continue
            patient_folder = os.path.join(data_directory, patient_id)
            subfolders = sorted([f.name for f in os.scandir(patient_folder) if f.is_dir() and not f.name.startswith('.')])
            for study_id in subfolders:
                unique_id = f"{patient_id}_{study_id}"
                # if checkpoint and (checkpoint.get('patient_id') == patient_id and checkpoint.get('study_id') >= study_id):
                #     logger.info(f"Skipping already processed combination {unique_id}")
                #     continue
                
                logger.info(f"-------------------------Processing patient {patient_id}-----------------------------")
                subfolder_path = os.path.join(patient_folder, study_id)
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                    coordinates = process_patient_folder(subfolder_path=subfolder_path)
                logger.info(f'Finished processing patient {patient_id}')
                suv_data = coordinates['suv_data']
                ct_data = coordinates['ct_data']

                positive_coordinates = coordinates['positive_coordinates']
                negative_coordinates = coordinates['negative_coordinates']
                non_restricted_pos_coordinates = coordinates['non_restricted_positive_coordinates']
                if modality == 'pet':
                    modality_data = suv_data
                elif modality == 'ct':
                    modality_data = ct_data
        
                logger.info(f'getting positive tumor coord')
                if len(positive_coordinates) != 0:
                    for idx, coordinate in enumerate(positive_coordinates):
                        # row_data = [patient_id, study_id, idx] + [str(coordinate)] + [str(block_size)] + [1]
                        suv_block = np.array(modality_data[slice(coordinate[0], coordinate[0] + block_size[0]),
                                       slice(coordinate[1], coordinate[1] + block_size[1]),
                                       slice(coordinate[2], coordinate[2] + block_size[2])])
                        raw_input_data.append(suv_block)
                        # raw_input_label.append(1)
            
                logger.info(f'getting non restricted positive tumor coord')
                if len(non_restricted_pos_coordinates) != 0:
                    for idx, coordinate in enumerate(non_restricted_pos_coordinates):
                        suv_block = np.array(modality_data[slice(coordinate[0], coordinate[0] + block_size[0]),
                                       slice(coordinate[1], coordinate[1] + block_size[1]),
                                       slice(coordinate[2], coordinate[2] + block_size[2])])
                        raw_input_data.append(suv_block)
                        # raw_input_label.append(1)
    
                        # coordinate_str = f"({coordinate[0]},{coordinate[1]},{coordinate[2]})"
                        # block_size_str = f"({block_size[0]},{block_size[1]},{block_size[2]})"
                        # # Prepare the row data
                        # row_data = [patient_id, study_id, idx, coordinate_str, block_size_str, 2]
                        # csvwriter.writerow(row_data)
                logger.info(f'getting negative tumor coord')
                for idx, coordinate in enumerate(negative_coordinates):
                    suv_block = np.array(modality_data[slice(coordinate[0], coordinate[0] + block_size[0]),
                                       slice(coordinate[1], coordinate[1] + block_size[1]),
                                       slice(coordinate[2], coordinate[2] + block_size[2])])
                    raw_input_data.append(suv_block)
                    # raw_input_label.append(0)
                logger.info(f'Finished getting coord for study {study_id} for patient {patient_id} to npy file')
            # break

    print(f'Saving data')
    # print(f'length of raw_input_data: {len(raw_input_data)}')
    # print(f'shape of raw_input_data: {raw_input_data[0]}')
    # Save list as a pickle file
    with open('raw_input_ct.pkl', 'wb') as f:
        pickle.dump(raw_input_data, f)

    raw_output_data = np.array(raw_input_data, dtype=object)
    # raw_output_label = np.array(raw_input_label, dtype=int)
    if modality == 'pet':
        np.save(f'ori_reso_all_dataset.npy', raw_output_data)
    elif modality == 'ct':
        np.save(f'400_by_400_ct_dataset.npy', raw_output_data)
    else:
        np.save(f'ori_reso_all_dataset_some_modality.npy', raw_output_data)
    # np.save(f'ori_reso_all_labels.npy', raw_input_label)     
    print(f'data saved')      

def main(data_directory, modality, block_size = (3, 3, 3)):
    get_coorindates(data_directory,modality=modality, block_size = block_size)

if __name__ == "__main__":
    # data_directory = "/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data"
    print('start!')
    data_directory = "/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions"
    modality = 'ct'
    main(data_directory, modality, block_size=(3, 3, 3))
    print('done!')