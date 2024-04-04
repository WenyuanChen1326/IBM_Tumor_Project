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
import cc3d



logging.basicConfig(
    filename='get_all_pos_cc_pixel_count.log', 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()
def check_coord_boundaries(coord,block_size, shape):
    # Calculate the center coordinates
    x_center = coord[0]
    y_center = coord[1]
    z_center = coord[2]

    # Calculate half sizes for each dimension, adjusting for even sizes
    half_size_x = block_size[0] // 2
    half_size_y = block_size[1] // 2
    half_size_z = block_size[2] // 2

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
def process_patient_folder(subfolder_path, block_size = (3, 3, 3), min_distance = 3, threshold_abs = 2.0):
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
    labels_out = cc3d.connected_components(seg_data, connectivity = 26)
    # Find the indices where the value is 1
    # coordinates = np.where(seg_data == 1)
    # # The returned coordinates are in the form of (x_indices, y_indices, z_indices)
    # # To get a list of (x, y, z) tuples:
    # pos_xyz_coordinates = np.array(list(zip(coordinates[0], coordinates[1], coordinates[2])))
    # # use the same sampling method to get the pos and neg. 
    # intersection_pos = intersection_2d(local_max, pos_xyz_coordinates)
    # tuples_local_max = set(map(tuple, intersection_pos))
    # tuples_pos_xyz = set(map(tuple, pos_xyz_coordinates))
    # # Find the difference between the two sets
    # non_intersection_set = tuples_pos_xyz.difference(tuples_local_max)
    # non_intersection_points_neg = np.array(list(non_intersection_set))
    # assert len(non_intersection_points_neg) + len(intersection_pos) == len(local_max), f"len(non_intersection_points) + len(intersection_pos) != len(pos_xyz_coordinates)"
    return {
        'suv_data': suv_data,
        'seg_data': seg_data,
        'local_max': local_max,
        'connected_components': labels_out
    }
def get_coorindates(data_directory, block_size = (3, 3, 3), min_distance = 3, threshold_abs = 2.0):
    patients = sorted([pid for pid in os.listdir(data_directory) if not pid.startswith('.') and os.path.isdir(os.path.join(data_directory, pid))])
    non_intersection_points_neg_suv_block_lst = [] 
    intersection_pos_suv_block_lst = []
    raw_input_label_lst = []
    # Define the CSV file name
    csv_file_name = f'studies_summary_R{min_distance}_TH_{threshold_abs}_cc.csv'

    # Open the CSV file
    with open(csv_file_name, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        
        # Write the header if the file is empty/new
        if file.tell() == 0:
            csv_writer.writerow(['Patient ID', 'Study ID', 'Max CC','Unique CC Count', 'Total CC Count', 'CC List Count'])
  
        for patient_id in tqdm(patients, desc="Processing patients"):
            if patient_id == 'PETCT_1285b86bea':
                continue
            patient_folder = os.path.join(data_directory, patient_id)
            subfolders = sorted([f.name for f in os.scandir(patient_folder) if f.is_dir() and not f.name.startswith('.')])
            for study_id in subfolders:
                cc_lst = []
                unique_id = f"{patient_id}_{study_id}"
                # if checkpoint and (checkpoint.get('patient_id') == patient_id and checkpoint.get('study_id') >= study_id):
                #     logger.info(f"Skipping already processed combination {unique_id}")
                #     continue
                
                logger.info(f"-------------------------Processing patient {patient_id}-----------------------------")
                subfolder_path = os.path.join(patient_folder, study_id)
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                    coordinates = process_patient_folder(subfolder_path=subfolder_path, block_size = block_size, min_distance=min_distance, threshold_abs=threshold_abs)
                logger.info(f'Finished processing patient {patient_id}')
                suv_data = coordinates['suv_data']
                seg_data = coordinates['seg_data']
                candidate_coordinate = coordinates['local_max']
                connected_components = coordinates['connected_components']
                max_cc = np.max(connected_components)
                for idx, coordinate in enumerate(candidate_coordinate):
                    x, y, z = coordinate
                    component_label = connected_components[x, y, z]
                    if component_label > 0:
                        cc_lst.append(component_label)

                    # checked_coord = check_coord_boundaries(coordinate, block_size, suv_data.shape)
                    # suv_block = np.array(suv_data[slice(checked_coord[0], checked_coord[0] + block_size[0]),
                    #             slice(checked_coord[1], checked_coord[1] + block_size[1]),
                    #             slice(checked_coord[2], checked_coord[2] + block_size[2])])
                    # seg_block = np.array(seg_data[slice(checked_coord[0], checked_coord[0] + block_size[0]),
                    #             slice(checked_coord[1], checked_coord[1] + block_size[1]),
                    #             slice(checked_coord[2], checked_coord[2] + block_size[2])])
                    # if np.all(seg_block == 0):
                    #     non_intersection_points_neg_suv_block_lst.append(suv_block)
                    #     raw_input_label_lst.append(0)
                    #     neg_counter +=1
                    # else:
                    #     intersection_pos_suv_block_lst.append(suv_block)
                    #     raw_input_label_lst.append(1)
                    #     pos_counter +=1



                # logger.info(f'getting positive tumor coord')
                # logger.info(f'Finished getting coord for study {study_id} for patient {patient_id} to npy file')
                # pos_coord_length = pos_counter
                # neg_coord_length = neg_counter
                # total_coord_length = len(candidate_coordinate)
                csv_writer.writerow([patient_id, study_id, max_cc,len(set(cc_lst)), len(cc_lst), str(cc_lst)])

                logger.info(f'Written lengths for study {study_id} for patient {patient_id} to CSV file')

            # break

    # print(f'Saving data')
    # # print(f'length of raw_input_data: {len(raw_input_data)}')
    # # print(f'shape of raw_input_data: {raw_input_data[0]}')
    # # Save list as a pickle file
    # with open(f'intersection_pos_suv_block_lst_R_{min_distance}_TH_{threshold_abs}', 'wb') as f:
    #     pickle.dump(intersection_pos_suv_block_lst, f)
    # with open(f'non_intersection_points_neg_suv_block_lst_R{min_distance}_TH_{threshold_abs}', 'wb') as f:
    #     pickle.dump(non_intersection_points_neg_suv_block_lst, f)
    # with open(f'raw_input_label_lst_R{min_distance}_TH_{threshold_abs}', 'wb') as f:
    #     pickle.dump(raw_input_label_lst, f)   
    # print(f'data saved')      

def main(data_directory, block_size = (3, 3, 3), min_distance = 3, threshold_abs = 2.0):
    get_coorindates(data_directory, block_size = block_size, min_distance = min_distance, threshold_abs = threshold_abs)
if __name__ == "__main__":
    # data_directory = "/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data"
    print('start!')
    data_directory = "/Volumes/T7 Shield/FDG-PET-CT-Lesions"
    main(data_directory, block_size = (3, 3, 3), min_distance = 1, threshold_abs = 2.0)
    print('done!')






