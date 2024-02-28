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


logging.basicConfig(
    filename='patient_processing_candidates_blocks_sample_size_100.log', 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger()
def create_negative_blocks(seg_data, local_max,sample_size, block_size):
    negative_coordinates = sample_neg_block(seg_data,local_max,block_size, sample_size)
    return negative_coordinates

def create_positive_blocks(filtered_separate_segmentation_masks):
    positive_coordinates = get_tumor_under_threshold_blocks_per_study(filtered_separate_segmentation_masks, block_size = (3, 3, 3))
    return positive_coordinates

def process_patient_folder(subfolder_path, sample_size = 100, block_size = (3, 3, 3)):
    seg_file_path = os.path.join(subfolder_path, 'SEG.nii.gz')
    suv_file_path = os.path.join(subfolder_path, 'SUV.nii.gz')

    seg_img = nib.load(seg_file_path)
    seg_data = seg_img.get_fdata()
    voxel_dimensions = seg_img.header['pixdim'][1:4]

    suv_img = nib.load(suv_file_path)
    suv_data = suv_img.get_fdata()

    separate_segmentation_masks = get_connected_components_3D(seg_data)
    filtered_separate_segmentation_masks = filter_separate_segmentation_mask_by_diameter_and_SUV_max_and_voxel_of_interest(
        suv_data, voxel_dimensions, separate_segmentation_masks, 
        diameter_in_cm = 0.6, SUV_max = 3, voxel_of_interst = 3)
    R=3
    TH=2.0
    local_max=peak_local_max(suv_data,min_distance=R,threshold_abs=TH)
    positive_coordinates = create_positive_blocks(filtered_separate_segmentation_masks)
    negative_coordinates = create_negative_blocks(seg_data,local_max,sample_size, block_size)
    return {
        'positive_coordinates': positive_coordinates,
        'negative_coordinates': negative_coordinates
    }


def write_to_csv(data_directory, output_csv, block_size = (3, 3, 3)):
    # checkpoint = load_checkpoint()
    patients = sorted([pid for pid in os.listdir(data_directory) if not pid.startswith('.') and os.path.isdir(os.path.join(data_directory, pid))])

    with open(output_csv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header_written = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0

        if not header_written:
            header = ['Patient ID', 'Study ID', 'Tumor idx', 'Coordinate', 'Block Size', 'Positive Tumor']
            csvwriter.writerow(header)

        for patient_id in tqdm(patients, desc="Processing patients"):
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
                # If no tumors, write a row with zeros for the tumor-specific columns
                logger.info(f'start writing this study to csv')
                positive_coordinates = coordinates['positive_coordinates']
                negative_coordinates = coordinates['negative_coordinates']
                if len(positive_coordinates) == 0:
                    row_data = [patient_id, study_id, 0] + [0]*3
                    csvwriter.writerow(row_data)
                else:
                    for idx, coordinate in enumerate(positive_coordinates):
                        # row_data = [patient_id, study_id, idx] + [str(coordinate)] + [str(block_size)] + [1]
                        coordinate_str = f"({coordinate[0]},{coordinate[1]},{coordinate[2]})"
                        block_size_str = f"({block_size[0]},{block_size[1]},{block_size[2]})"
                        # Prepare the row data
                        row_data = [patient_id, study_id, idx, coordinate_str, block_size_str, 1]
                        csvwriter.writerow(row_data)
                for idx, coordinate in enumerate(negative_coordinates):
                    coordinate_str = f"({coordinate[0]},{coordinate[1]},{coordinate[2]})"
                    block_size_str = f"({block_size[0]},{block_size[1]},{block_size[2]})"
                    # row_data = [patient_id, study_id, idx] +str(coordinate) + str(block_size) + [0]
                    row_data = [patient_id, study_id, idx, coordinate_str, block_size_str, 0]
                    csvwriter.writerow(row_data)
                logger.info(f'Finished writing study {study_id} for patient {patient_id} to csv')
                

def main(data_directory, output_csv, block_size = (3, 3, 3)):
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['Patient ID', 'Study ID', 'Tumor idx', 'Coordinate', 'Block Size', 'Positive Tumor']
        csvwriter.writerow(header)
    write_to_csv(data_directory, output_csv, block_size=block_size)

if __name__ == "__main__":
    # data_directory = "/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data"
    print('start!')
    data_directory = "/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions"
    output_csv = 'all_patients_tumor_pos_neg_block_sample_size_100.csv'
    main(data_directory, output_csv)
    print('done!')