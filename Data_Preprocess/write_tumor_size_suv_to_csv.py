import os
import numpy as np
import nibabel as nib
import csv
from utils import *
from tqdm import tqdm

def process_patient_folder(patient_folder):
    # Filter out any non-directory entries, like .DS_Store
    subfolders = [f.name for f in os.scandir(patient_folder) if f.is_dir()]
    if not subfolders:
        raise ValueError(f"No subdirectories found in {patient_folder}")
    subfolder_name = subfolders[0]  # Get the first subdirectory
    subfolder_path = os.path.join(patient_folder, subfolder_name)
    seg_file_path = os.path.join(subfolder_path, 'SEG.nii.gz')
    suv_file_path = os.path.join(subfolder_path, 'SUV.nii.gz')

    seg_img = nib.load(seg_file_path)
    seg_data = seg_img.get_fdata()
    voxel_dimensions = seg_img.header['pixdim'][1:4]

    suv_img = nib.load(suv_file_path)
    suv_data = suv_img.get_fdata()

    separate_seg_masks = get_connected_components_3D(seg_data)
    tumor_voxel_counts, tumor_volumes = calculate_tumor_volumes(separate_seg_masks, voxel_dimensions)
    SUV_values = calculate_SUV_value(separate_seg_masks, suv_data)

    return tumor_voxel_counts, tumor_volumes, SUV_values

def get_tumor_stats(tumor_volumes, SUV_values):
    max_volume = np.max(tumor_volumes)
    min_volume = np.min(tumor_volumes)
    mean_volume = np.mean(tumor_volumes)

    max_suv = np.max(SUV_values)
    min_suv = np.min(SUV_values)
    mean_suv = np.mean(SUV_values)
    return (max_volume, min_volume, mean_volume, max_suv, min_suv, mean_suv)

def write_to_csv(data_directory, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header_written = False

        for patient_id in tqdm(os.listdir(data_directory), desc = "Processing patients"):
            patient_folder = os.path.join(data_directory, patient_id)
            if os.path.isdir(patient_folder):
                print(f"Processing patient {patient_id}")
                voxel_counts, volumes, suvs = process_patient_folder(patient_folder)
                print(f"Voxel Counts: {voxel_counts}")
                print(f"Volumes: {volumes}")
                print(f"SUVs: {suvs}")
                # only calculate stats if there are any tumors
                stats = get_tumor_stats(volumes, suvs) if volumes else (0, 0, 0, 0, 0, 0)

                # Prepare row data with stats
                row_data = [patient_id]
                for idx, (volume, suv) in enumerate(zip(volumes, suvs)):
                    row_data.extend([volume, suv])
                row_data.extend(stats)  # Append stats to the row

                # Write header if not already done
                if not header_written:
                    header = ['Patient ID']
                    for i in range(1, len(volumes) + 1):
                        header.extend([f'Tumor {i} Volume (cubic mm)', f'Tumor {i} SUV Value'])
                    # Add headers for stats
                    header.extend(['Max Volume (cubic mm)', 'Min Volume (cubic mm)', 'Mean Volume (cubic mm)',
                                   'Max SUV', 'Min SUV', 'Mean SUV'])
                    csvwriter.writerow(header)
                    header_written = True

                # Write row data
                csvwriter.writerow(row_data)
def main(data_directory, output_csv):
    write_to_csv(data_directory, output_csv)


if __name__ == "__main__":
    data_directory =  "/Users/wenyuanchen/Desktop/IBM/Data"
    output_csv = 'results.csv'
    main(data_directory, output_csv)



