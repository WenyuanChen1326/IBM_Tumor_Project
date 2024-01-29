import numpy as np
from scipy import ndimage
import nibabel as nib

def get_connected_components_3D(seg_data):
    # input seg_data is the numpy after reading nifti and get_fdata()
   
    #value of 1 means lesion is present
    value = 1
    binary_mask = seg_data == value

    #label and seperate each component off that has the value of 1
    labled_mask, num_features = ndimage.label(binary_mask)

    #assign a unique id to each component
    unique_ids = np.unique(labled_mask)[1:]
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
        tumor_volume = num_voxels * np.prod(voxel_dimensions)
        # print(f"Tumor volume: {tumor_volume} cubic mm")
        tumor_volumes.append(tumor_volume)
        # break
    return tumor_voxel_counts, tumor_volumes

def calculate_SUV_value(separate_seg_masks, SUV_data):
    SUV_values = []
    for mask in separate_seg_masks:
        # apply the mask to the SUV data
        tumor_suv_data = SUV_data[mask]
        # calculate the SUV value within the masked region
        SUV_value = np.sum(tumor_suv_data)
        # print(f"Tumor volume: {tumor_volume} cubic mm")
        SUV_values.append(SUV_value)
    return SUV_values
def main(seg_data, SUV_data, voxel_dimensions):
    separate_seg_masks = get_connected_components_3D(seg_data)
    tumor_voxel_counts, tumor_volumes = calculate_tumor_volumes(separate_seg_masks, voxel_dimensions)
    SUV_values = calculate_SUV_value(separate_seg_masks, SUV_data)
    for idx, volume in enumerate(tumor_volumes):
        print(f"Tumor {idx+1}: Volume = {volume} cubic mm")
    for idx, voxel_count in enumerate(tumor_voxel_counts):
        print(f"Tumor {idx+1}: Voxel Count = {voxel_count}")
    for idx, value in enumerate(SUV_values):
        print(f"Tumor {idx+1}: SUV Value = {value}")
    return tumor_voxel_counts, tumor_volumes, SUV_values

if __name__ == "__main__":
    seg_file_path = '/Users/wenyuanchen/Desktop/IBM/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SEG.nii.gz'
    seg_img = nib.load(seg_file_path)
    # Convert the image data to a numpy array
    seg_data = seg_img.get_fdata()
    suv_file_path = '/Users/wenyuanchen/Desktop/IBM/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SUV.nii.gz'
    suv_img = nib.load(suv_file_path)
    # Convert the image data to a numpy array
    suv_data = suv_img.get_fdata()
    voxel_dimensions = seg_img.header['pixdim'][1:4]
    main(seg_data=seg_data, SUV_data=suv_data, voxel_dimensions=voxel_dimensions)
