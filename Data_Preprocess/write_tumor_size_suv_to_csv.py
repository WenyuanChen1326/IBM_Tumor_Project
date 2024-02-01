import os
import numpy as np
import nibabel as nib
import csv
from utils import *
from petct_utils import *
from tqdm import tqdm
import contextlib
import logging
import json

CHECKPOINT_FILE = 'checkpoint.json'

# def save_checkpoint(processed_patients):
#     with open(CHECKPOINT_FILE, 'w') as f:
#         json.dump(processed_patients, f)

# def load_checkpoint():
#     if os.path.exists(CHECKPOINT_FILE):
#         with open(CHECKPOINT_FILE, 'r') as f:
#             return json.load(f)
#     return []

def save_checkpoint(last_processed):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(last_processed, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_next_unprocessed(patients, last_processed):
    if not last_processed:
        return patients[0], 0  # Start from the beginning if no last processed
    try:
        last_patient_idx = patients.index(last_processed['patient_id'])
        if last_patient_idx + 1 < len(patients):
            return patients[last_patient_idx + 1], last_patient_idx + 1  # Next patient
    except ValueError:
        return patients[0], 0  # If last processed patient not found, start from beginning
    return None, None  # If we've processed everything

# Set up logging
logging.basicConfig(
    filename='patient_processing_with_26_conn.log', 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger()

def process_patient_folder(subfolder_path):
    seg_file_path = os.path.join(subfolder_path, 'SEG.nii.gz')
    suv_file_path = os.path.join(subfolder_path, 'SUV.nii.gz')

    seg_img = nib.load(seg_file_path)
    seg_data = seg_img.get_fdata()
    voxel_dimensions = seg_img.header['pixdim'][1:4]

    suv_img = nib.load(suv_file_path)
    suv_data = suv_img.get_fdata()

    separate_seg_masks = get_connected_components_3D(seg_data, connectivity=26)
    tumor_voxel_counts, tumor_volumes, tumor_in_diameter = calculate_tumor_volumes(separate_seg_masks, voxel_dimensions)
    SUV_mean_values, SUV_max_values, SUV_min_values= calculate_SUV_value(separate_seg_masks, suv_data)
    # plot_the_tumor_idx_on_suv_data(subfolder_path, separate_seg_masks, suv_data)

    return {
        'separate_seg_masks': separate_seg_masks,
        'tumor_voxel_counts': tumor_voxel_counts,
        'tumor_volumes': tumor_volumes,
        'tumor_in_diameter': tumor_in_diameter,
        'SUV_mean_values': SUV_mean_values,
        'SUV_max_values': SUV_max_values,
        'SUV_min_values': SUV_min_values
    }
def plot_the_tumor_idx_on_suv_data(study_path, separate_seg_masks, SUV_data):
    slice_idx_lst = find_tumor_idx_on_suv_data(separate_seg_masks, SUV_data)
    for idx, slice_idx in enumerate(slice_idx_lst):
        logger.info(f"Plotting Tumor {idx +1}: Index on SUV data = {slice_idx}")
        read_and_show_study(study_path, idx=slice_idx, modality='pet_seg', out_path=f"{study_path}/pet_seg_slice{slice_idx}_tumor_idx{idx+1}_.png")

def write_to_csv(data_directory, output_csv):
    checkpoint = load_checkpoint()
    patients = sorted([pid for pid in os.listdir(data_directory) if not pid.startswith('.') and os.path.isdir(os.path.join(data_directory, pid))])

    with open(output_csv, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header_written = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0

        if not header_written:
            header = ['Patient ID', 'Study ID', 'Tumor idx', 'Pixel Vol', 'Physical Vol(cm^3)', 'In Diameter(cm)', 'SUV Mean', 'SUV Min', 'SUV Max']
            csvwriter.writerow(header)

        for patient_id in tqdm(patients, desc="Processing patients"):
            patient_folder = os.path.join(data_directory, patient_id)
            subfolders = sorted([f.name for f in os.scandir(patient_folder) if f.is_dir() and not f.name.startswith('.')])
            for study_id in subfolders:
                unique_id = f"{patient_id}_{study_id}"
                if checkpoint and (checkpoint.get('patient_id') == patient_id and checkpoint.get('study_id') >= study_id):
                    logger.info(f"Skipping already processed combination {unique_id}")
                    continue
                
                logger.info(f"-------------------------Processing patient {patient_id}-----------------------------")
                subfolder_path = os.path.join(patient_folder, study_id)
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                    stats = process_patient_folder(subfolder_path=subfolder_path)
                logger.info(f'Finished processing patient {patient_id}')
                # If no tumors, write a row with zeros for the tumor-specific columns
                logger.info(f'start writing this study to csv')
                if not stats['tumor_voxel_counts']:
                    row_data = [patient_id, study_id, 0] + [0]*6
                    csvwriter.writerow(row_data)
                else:
                    # Write a row for each tumor
                    for idx in range(len(stats['tumor_voxel_counts'])):
                        row_data = [patient_id, study_id, idx+1]
                        row_data.extend([
                            stats['tumor_voxel_counts'][idx],
                            stats['tumor_volumes'][idx],
                            stats['tumor_in_diameter'][idx],
                            stats['SUV_mean_values'][idx],
                            stats['SUV_min_values'][idx],
                            stats['SUV_max_values'][idx]
                        ])
                        csvwriter.writerow(row_data)
                save_checkpoint({'patient_id': patient_id, 'study_id': study_id})
                logger.info(f'Finished writing study {study_id} for patient {patient_id} to csv')
                
            

def main(data_directory, output_csv):

    if not os.path.exists(CHECKPOINT_FILE):
        # If there's no checkpoint file, write the header to a new CSV file
        with open(output_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = ['Patient ID', 'Study ID', 'Tumor idx', 'Pixel Vol', 'Physical Vol(cm^3)', 'In Diameter(cm)', 'SUV Mean', 'SUV Min', 'SUV Max']
            csvwriter.writerow(header)
    write_to_csv(data_directory, output_csv)

if __name__ == "__main__":
    # data_directory = "/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data"
    print('start!')
    data_directory = "/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions"
    output_csv = 'all_patients_results_with_26.csv'
    main(data_directory, output_csv)
    print('done!')



