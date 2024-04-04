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
    filename='get_region_suv_block.log', 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()
def process_patient_folder(subfolder_path, sample_size = 100, block_size = (3, 3, 3), diameter_in_cm = 0.6, SUV_max = 3, voxel_of_interst = 3):
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
    diameter_in_cm = diameter_in_cm, SUV_max = SUV_max, voxel_of_interst = voxel_of_interst )
    suv_block_lst = []
    voxel_size_lst = []
    for mask in filtered_separate_segmentation_masks:
        coordinates = np.where(mask == 1)
        voxel_size = np.sum(mask)
        pos_xyz_coordinates = np.array(list(zip(coordinates[0], coordinates[1], coordinates[2])))
        non_restricted_positive_coordinates = sample_neg_block(seg_data,pos_xyz_coordinates,block_size, sample_size, negative=False)
        for idx, coordinate in enumerate(non_restricted_positive_coordinates):
            suv_block = np.array(suv_data[slice(coordinate[0], coordinate[0] + block_size[0]),
                        slice(coordinate[1], coordinate[1] + block_size[1]),
                        slice(coordinate[2], coordinate[2] + block_size[2])])
            suv_block_lst.append(suv_block)
            voxel_size_lst.append(voxel_size)

    return suv_block_lst, voxel_size_lst

def get_region_coorindates(data_directory, sample_size = 10, block_size = (3, 3, 3), diameter_in_cm = 0.6, SUV_max = 3, voxel_of_interst = 3):
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
            # print(subfolder_path)
            # with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            suv_block_lst, voxel_size_lst = process_patient_folder(subfolder_path=subfolder_path, sample_size = sample_size, block_size = block_size, diameter_in_cm = diameter_in_cm , SUV_max = SUV_max, voxel_of_interst = voxel_of_interst)
            one_patient_output = {
                'patient_id': patient_id,
                'study_id': study_id,
                'suv_block_lst': suv_block_lst,
                'voxel_size_lst': voxel_size_lst}
            final_output.append(one_patient_output)
            logger.info(f"-------------------------Finished processing patient {patient_id}-----------------------------")

    with open(f"pos_region_with_size_vox_{voxel_of_interst}_sample_size_{sample_size}.pkl", "wb") as file:
    # Dump the list of dictionaries to the file
        pickle.dump(final_output, file)
def main(data_directory, args):
    get_region_coorindates(data_directory, sample_size = args.sample_size, block_size = args.block_size, diameter_in_cm = args.diameter_in_cm, SUV_max = args.SUV_max, voxel_of_interst = args.voxel_of_interst)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get region suv block')
    parser.add_argument('--sample_size', type=int, default= 5, help='The number of sample size')
    parser.add_argument('--block_size', type=tuple, default=(3, 3, 3), help='The block size')
    parser.add_argument('--diameter_in_cm', type=float, default=0.6, help='The diameter in cm')
    parser.add_argument('--SUV_max', type=float, default=3, help='The SUV max')
    parser.add_argument('--voxel_of_interst', type=float, default=np.inf, help='The voxel of interest')
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    # data_directory = "/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data"
    logger.info('start!')
    # data_directory = "/Volumes/T7 Shield/FDG-PET-CT-Lesions"

    # main(data_directory, args)
    logger.info('done!')

    with open("/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/pos_region_with_size_vox_inf_sample_size_5.pkl", "rb") as file:
    # Load the list of dictionaries from the file
        loaded_data = pickle.load(file)
    # Print the loaded data
    print(len(loaded_data))
    print((loaded_data[1]))
