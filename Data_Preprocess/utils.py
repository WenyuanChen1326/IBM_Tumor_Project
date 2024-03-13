import numpy as np
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import math
import cc3d
import SimpleITK as sitk

def get_connected_components_3D(seg_data, connectivity =26):
    # print(seg_data.shape)
    # 4 is a rough estimate of the minimum volume of a tumor in cubic mm
    # minV= calculate_sphere_vol_from_diameter(4) #(removing noise in the segmentation data but not sure what the best number of pixel to deem too small is)
    labels_out = cc3d.connected_components(seg_data, connectivity = connectivity)

    # print(f"labels_out shape: {labels_out.shape}")
    cc_n = np.max(np.unique(labels_out))
    separate_seg_masks = []
    for i in range(1,cc_n+1):
        # print(f"tumor index: {i}")
        # size_n=np.sum(labels_out==i)
        # if size_n<minV:
        #     seg_data[labels_out==i]=0
        # else:
        c_mask = labels_out == i
        separate_seg_masks.append(c_mask)
    return separate_seg_masks

# def calculate_sphere_vol_from_diameter(diameter):
#     radius = diameter/2
#     return (4/3)*np.pi*(radius**3)

# def get_connected_components_3D(seg_data, connectivity = 26):
#     # input seg_data is the numpy after reading nifti and get_fdata()
#     #value of 1 means lesion is present
#     value = 1
#     binary_mask = seg_data == value
#     print(f"binary_mask shape: {binary_mask.shape}")

#     #label and seperate each component off that has the value of 1
#     if connectivity == 18:
#         connectivity_criteria = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
#                          [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
#                          [[0, 1, 0], [1, 1, 1], [0, 1, 0]]])
#     else:
#         connectivity_criteria = np.ones((3, 3, 3), dtype=int)

#     labled_mask, num_features = ndimage.label(binary_mask, structure=connectivity_criteria)

#     #assign a unique id to each component
#     unique_ids = np.unique(labled_mask)[1:]
#     # unique_ids = np.unique(labled_mask)
#     # TO-DO: can we slice the unique_ids to exclude the first one, which is 0 corresponding to the background?
#     # print(unique_ids)

#     #num of components
#     print(num_features)

#     #seperate out the masks
#     separate_seg_masks = []
#     for component_id in unique_ids:
#         component_mask = labled_mask == component_id
#         #print each id of each component
#         print(f"Connected Component {component_id}:")
#         if np.sum(component_mask) >= calculate_sphere_vol_from_diameter(4):
#             print(calculate_sphere_vol_from_diameter(4))
#             separate_seg_masks.append(component_mask)
#     return separate_seg_masks

def get_diameter_from_sphere_volume(volume):
    return 2 * (3 * volume / (4 * np.pi)) ** (1/3) 


def calculate_tumor_volumes(separate_seg_masks, voxel_dimensions):
    tumor_volumes = []
    tumor_voxel_counts = []
    tumor_in_diameter = []
    for mask in separate_seg_masks:
        #count the number of voxels in the mask
        num_voxels = np.sum(mask)
        tumor_voxel_counts.append(num_voxels)
        #the volume of each voxel is the product of its dimensions
        tumor_volume = num_voxels * np.prod(voxel_dimensions)/1000
        # print(f"Tumor volume: {tumor_volume} cubic mm")
        tumor_volumes.append(np.round(tumor_volume,4))
        tumor_in_diameter.append(np.round(get_diameter_from_sphere_volume(tumor_volume),4))
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

# def scale_up_block(input_block, new_resol = [224,224,224],interpolation = 'cubic', spacing = (2.0364201068878174, 2.0364201068878174, 3.0)):
#     # Convert the input numpy array to a SimpleITK Image
#     input_image = sitk.GetImageFromArray(input_block.astype(np.float32))
#     # Define the new size
#     new_size = new_resol
#     # Compute the scaling factors for each dimension
#     scaling_factors = [float(new_size[0]) / input_image.GetSize()[0], 
#                        float(new_size[1]) / input_image.GetSize()[1], 
#                        float(new_size[2]) / input_image.GetSize()[2]]
#     # Create a resampling filter
#     resample_filter = sitk.ResampleImageFilter()
#     # Set the output image size
#     resample_filter.SetSize(new_size)

#     # Set the interpolator. 
#     if interpolation == 'cubic':
#         resample_filter.SetInterpolator(sitk.sitkBSpline)
#     # For cubic interpolation, change to sitk.sitkBSpline.
#     if interpolation == 'linear':
#         resample_filter.SetInterpolator(sitk.sitkLinear)
#     # Calculate new spacing based on the scaling factors
#     # original_spacing = input_image.GetSpacing()
#     original_spacing = spacing
#     new_spacing = [original_spacing[0] / scaling_factors[0], 
#                    original_spacing[1] / scaling_factors[1], 
#                    original_spacing[2] / scaling_factors[2]]
#     resample_filter.SetOutputSpacing(new_spacing)
#     # Set the output origin, to keep it same as input
#     resample_filter.SetOutputOrigin(input_image.GetOrigin())

#     # Set the output direction, to keep it same as input
#     resample_filter.SetOutputDirection(input_image.GetDirection())

#     # Perform the resampling
#     resampled_image = resample_filter.Execute(input_image)

#     # Convert the resampled SimpleITK image to a NumPy array
#     resampled_array = sitk.GetArrayFromImage(resampled_image)
#     # print(resampled_array.shape)

#     # Transpose the array to match the conventional (height, width, channels) format
#     # resampled_array_np = np.transpose(resampled_array, (1, 2, 0))
#     # print(resampled_array_np.shape)

#     # Return the resampled array
#     return resampled_array

def scale_up_block(input_block, new_resol = [224,224,224],interpolation = 'cubic', spacing = (2.0364201068878174, 2.0364201068878174, 3.0)):
    # Convert the input numpy array to a SimpleITK Image
    input_image = sitk.GetImageFromArray(input_block.astype(np.float32))
    # Define the new size
    new_size = new_resol
    # Compute the scaling factors for each dimension
    scaling_factors = [float(new_size[0]) / input_image.GetSize()[0], 
                       float(new_size[1]) / input_image.GetSize()[1], 
                       float(new_size[2]) / input_image.GetSize()[2]]
    # Create a resampling filter
    resample_filter = sitk.ResampleImageFilter()
    # Set the output image size
    resample_filter.SetSize(new_size)

    # Set the interpolator. 
    if interpolation == 'cubic':
        resample_filter.SetInterpolator(sitk.sitkBSpline)
    # For cubic interpolation, change to sitk.sitkBSpline.
    if interpolation == 'linear':
        resample_filter.SetInterpolator(sitk.sitkLinear)
    # Calculate new spacing based on the scaling factors
    # original_spacing = input_image.GetSpacing()
    original_spacing = spacing
    new_spacing = [original_spacing[0] / scaling_factors[0], 
                   original_spacing[1] / scaling_factors[1], 
                   original_spacing[2] / scaling_factors[2]]
    resample_filter.SetOutputSpacing(new_spacing)
    # Set the output origin, to keep it same as input
    resample_filter.SetOutputOrigin(input_image.GetOrigin())

    # Set the output direction, to keep it same as input
    resample_filter.SetOutputDirection(input_image.GetDirection())

    # Perform the resampling
    resampled_image = resample_filter.Execute(input_image)

    # Convert the resampled SimpleITK image to a NumPy array
    resampled_array = sitk.GetArrayFromImage(resampled_image)
    # print(resampled_array.shape)

    # Transpose the array to match the conventional (height, width, channels) format
    resampled_array_np = np.transpose(resampled_array, (1, 2, 0))
    # print(resampled_array_np.shape)

    # Return the resampled array
    # return resampled_array
    return resampled_array_np

    


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
 

