from utils import get_connected_components_3D
import numpy as np
from utils import *
from tqdm import tqdm
from create_pos_block import *
from random_sampling_neg_block import *
import csv
import os
import contextlib
import argparse
import logging
from skimage.feature import peak_local_max
import pickle

logging.basicConfig(
    filename='get_neg_high_suv_block.log', 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

def process_patient_folder(subfolder_path, sample_size = 10, block_size = (3, 3, 3), max_suv_min_threshold = 5.0):
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

    R=3
    TH= max_suv_min_threshold
    local_max=peak_local_max(suv_data,min_distance=R,threshold_abs=TH)
    
    negative_coordinates = sample_neg_block(seg_data,local_max, block_size = block_size, sample_size = sample_size)
    return {
        'suv_data': suv_data,
        'negative_coordinates': negative_coordinates
    }

def get_region_coorindates(data_directory, sample_size = 10, block_size = (3, 3, 3), max_suv_min_threshold = 5):
    patients = sorted([pid for pid in os.listdir(data_directory) if not pid.startswith('.') and os.path.isdir(os.path.join(data_directory, pid))])
    final_output = []
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
            # 
            logger.info(f"-------------------------Processing patient {patient_id}-----------------------------")
            subfolder_path = os.path.join(patient_folder, study_id)
            neg_output_suv_bock_list = []
            # print(subfolder_path)
            # with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            results = process_patient_folder(subfolder_path=subfolder_path, sample_size = sample_size, block_size = block_size, max_suv_min_threshold = max_suv_min_threshold)
            for idx, coordinate in enumerate(results['negative_coordinates']):
                suv_block = np.array(results['suv_data'][slice(coordinate[0], coordinate[0] + block_size[0]),
                            slice(coordinate[1], coordinate[1] + block_size[1]),
                            slice(coordinate[2], coordinate[2] + block_size[2])])
                assert np.max(suv_block ) >= max_suv_min_threshold, f" max_suv is less than {max_suv_min_threshold}"
                neg_output_suv_bock_list.append(suv_block)
            one_patient_output = {
                'patient_id': patient_id,
                'study_id': study_id,
                'suv_block_lst': neg_output_suv_bock_list}
            final_output.append(one_patient_output)
            logger.info(f"-------------------------Finished processing patient {patient_id}-----------------------------")

    with open(f"neg_region_max_suv_min_threshold_{max_suv_min_threshold}_sample_size_{sample_size}.pkl", "wb") as file:
    # Dump the list of dictionaries to the file
        pickle.dump(final_output, file)
def main(data_directory, args):
    get_region_coorindates(data_directory, sample_size = args.sample_size, block_size = args.block_size, max_suv_min_threshold = args.max_suv_min_threshold)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get region suv block')
    parser.add_argument('--sample_size', type=int, default= 10, help='The number of sample size')
    parser.add_argument('--block_size', type=tuple, default=(3, 3, 3), help='The block size')
    parser.add_argument('--max_suv_min_threshold', type=float, default=5, help='The SUV_max min threshold')
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    # data_directory = "/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data"
    logger.info('start!')
    # data_directory = "/Volumes/T7 Shield/FDG-PET-CT-Lesions"

    # main(data_directory, args)
    logger.info('done!')

    with open("/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/neg_region_max_suv_min_threshold_5_sample_size_10.pkl", "rb") as file:
    # Load the list of dictionaries from the file
        loaded_data = pickle.load(file)
    # Print the loaded data
    print(len(loaded_data))
    # print((loaded_data))
