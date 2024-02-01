import numpy as np
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import math

def get_connected_components_3D(seg_data, connectivity = 18):
    # input seg_data is the numpy after reading nifti and get_fdata()
    #value of 1 means lesion is present
    value = 1
    binary_mask = seg_data == value
    print(f"binary_mask shape: {binary_mask.shape}")

    #label and seperate each component off that has the value of 1
    if connectivity == 18:
        connectivity_criteria = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                         [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                         [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
    else:
        connectivity_criteria = np.ones((3, 3, 3), dtype=int)

    labled_mask, num_features = ndimage.label(binary_mask, structure=connectivity_criteria)

    #assign a unique id to each component
    unique_ids = np.unique(labled_mask)[1:]
    # unique_ids = np.unique(labled_mask)
    # TO-DO: can we slice the unique_ids to exclude the first one, which is 0 corresponding to the background?
    # print(unique_ids)

    #num of components
    print(num_features)

    #seperate out the masks
    separate_seg_masks = []
    for component_id in unique_ids:
        component_mask = labled_mask == component_id
        #print each id of each component
        print(f"Connected Component {component_id}:")
        # if np.sum(component_mask) >= 10:
        separate_seg_masks.append(component_mask)
    return separate_seg_masks

def calculate_tumor_volumes(separate_seg_masks, voxel_dimensions):
    tumor_volumes = []
    tumor_voxel_counts = []
    for mask in separate_seg_masks:
        #count the number of voxels in the mask
        num_voxels = np.sum(mask)
        tumor_voxel_counts.append(num_voxels)
        #the volume of each voxel is the product of its dimensions
        tumor_volume = num_voxels * np.prod(voxel_dimensions)/1000
        # print(f"Tumor volume: {tumor_volume} cubic mm")
        tumor_volumes.append(np.round(tumor_volume,4))
    tumor_in_diameter = np.round(np.cbrt(tumor_volumes),4)
    return tumor_voxel_counts, tumor_volumes, tumor_in_diameter

def calculate_SUV_value(separate_seg_masks, SUV_data):
    SUV_mean_values = []
    SUV_max_values =[]
    SUV_min_values = []
    for mask in separate_seg_masks:
        # apply the mask to the SUV data
        tumor_suv_data = SUV_data[mask]
        # calculate the SUV value within the masked region
        SUV_max = np.round(np.max(tumor_suv_data),4)
        SUV_min = np.round(np.min(tumor_suv_data),4)
        SUV_mean = np.round(np.mean(tumor_suv_data),4)
        # print(f"Tumor volume: {tumor_volume} cubic mm")
        SUV_max_values.append(SUV_max)
        SUV_min_values.append(SUV_min)
        SUV_mean_values.append(SUV_mean)
    return SUV_mean_values, SUV_max_values, SUV_min_values

def find_tumor_idx_on_suv_data(separate_seg_masks, SUV_data):
    tumor_idx_on_suv_data = []
    for mask in separate_seg_masks:
        maksed_SUV_data = SUV_data*mask
        # print(np.argmax(np.sum(SUV_data, axis = (0,1))))
        # print(f"maksed_SUV_data shape: {maksed_SUV_data.shape}")
        idx = np.argmax(np.sum(maksed_SUV_data, axis = (0,1)))
        print(f'tumor_idx: {idx}')
        tumor_idx_on_suv_data.append(idx)
    return tumor_idx_on_suv_data

def main(seg_data, SUV_data, voxel_dimensions, connectivity = 18):
    separate_seg_masks = get_connected_components_3D(seg_data, connectivity)
    tumor_voxel_counts, tumor_volumes, tumor_in_diameter = calculate_tumor_volumes(separate_seg_masks, voxel_dimensions)
    SUV_mean_values, SUV_max_values, SUV_min_values = calculate_SUV_value(separate_seg_masks, SUV_data)
    for idx, volume in enumerate(tumor_volumes):
        print(f"Tumor {idx}: Volume = {volume} cubic cm")
        print(f"Tumor {idx}: Cubic root Physical Volume = {np.round(math.cbrt(volume),2)} cubic cm")

    for idx, voxel_count in enumerate(tumor_voxel_counts):
        print(f"Tumor {idx}: Voxel Count = {voxel_count}")
    for idx, value in enumerate(SUV_mean_values):
        print(f"Tumor {idx}: SUV Mean Value = {value}")
    for idx, value in enumerate(SUV_max_values):
        print(f"Tumor {idx}: SUV Max Value = {value}")
    for idx, value in enumerate(SUV_min_values):
        print(f"Tumor {idx}: SUV Min Value = {value}")
    tumor_idx_on_suv_data = find_tumor_idx_on_suv_data(separate_seg_masks, SUV_data)
    print(f"tumor_idx_on_suv_data: {tumor_idx_on_suv_data}")
    return tumor_voxel_counts, tumor_volumes, SUV_mean_values, SUV_max_values, SUV_min_values, tumor_idx_on_suv_data

if __name__ == "__main__":
    # pet_file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/PET.nii.gz'
    seg_file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_1285b86bea/02-24-2006-NA-PET-CT Ganzkoerper  primaer mit KM-49419/SEG.nii.gz'
    # seg_file_path = '/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions/PETCT_1285b86bea/02-24-2006-NA-PET-CT Ganzkoerper  primaer mit KM-49419/SEG.nii.gz'
    seg_img = nib.load(seg_file_path)
    # Convert the image data to a numpy array
    seg_data = seg_img.get_fdata()
    suv_file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_1285b86bea/02-24-2006-NA-PET-CT Ganzkoerper  primaer mit KM-49419/SUV.nii.gz'
    # suv_file_path = '/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions/PETCT_1285b86bea/02-24-2006-NA-PET-CT Ganzkoerper  primaer mit KM-49419/SUV.nii.gz'
    print(f"working with {suv_file_path}")
    suv_img = nib.load(suv_file_path)
    # Convert the image data to a numpy array
    suv_data = suv_img.get_fdata()
    voxel_dimensions = seg_img.header['pixdim'][1:4]
    print(f'voxel_dimensions: {voxel_dimensions}')
    # print(f"working")
    main(seg_data=seg_data, SUV_data=suv_data, voxel_dimensions=voxel_dimensions, connectivity=26)
    # separate_seg_masks = get_connected_components_3D(seg_data)
    # # print(f'printing the mask 0')
    # print(seg_data.shape)
    # print (separate_seg_masks[0].shape)
 

