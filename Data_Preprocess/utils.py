import numpy as np
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import math

def get_connected_components_3D(seg_data):
    # input seg_data is the numpy after reading nifti and get_fdata()
   
    #value of 1 means lesion is present
    value = 1
    binary_mask = seg_data == value

    #label and seperate each component off that has the value of 1
    labled_mask, num_features = ndimage.label(binary_mask)

    #assign a unique id to each component
    # unique_ids = np.unique(labled_mask)[1:]
    unique_ids = np.unique(labled_mask)
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
        # break
    return tumor_voxel_counts, tumor_volumes

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


def main(seg_data, SUV_data, voxel_dimensions):
    separate_seg_masks = get_connected_components_3D(seg_data)
    tumor_voxel_counts, tumor_volumes = calculate_tumor_volumes(separate_seg_masks, voxel_dimensions)
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
    return tumor_voxel_counts, tumor_volumes, SUV_mean_values, SUV_max_values, SUV_min_values


if __name__ == "__main__":
    # pet_file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/PET.nii.gz'
    seg_file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_5d553bf6b4/09-16-2001-NA-PET-CT Ganzkoerper  primaer mit KM-78907/SEG.nii.gz'
    seg_img = nib.load(seg_file_path)
    # Convert the image data to a numpy array
    seg_data = seg_img.get_fdata()
    suv_file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_5d553bf6b4/09-16-2001-NA-PET-CT Ganzkoerper  primaer mit KM-78907/SUV.nii.gz'
    print(f"working with {suv_file_path}")
    suv_img = nib.load(suv_file_path)
    # Convert the image data to a numpy array
    suv_data = suv_img.get_fdata()
    voxel_dimensions = seg_img.header['pixdim'][1:4]
    print(f'voxel_dimensions: {voxel_dimensions}')
    # print(f"working")
    main(seg_data=seg_data, SUV_data=suv_data, voxel_dimensions=voxel_dimensions)
    # separate_seg_masks = get_connected_components_3D(seg_data)
    # # print(f'printing the mask 0')
    # print(seg_data.shape)
    # print (separate_seg_masks[0].shape)

