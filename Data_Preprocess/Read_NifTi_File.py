import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from utils import *
from display_removed_tumors import apply_color_coding_to_segmentation
from GIF_mip import create_mipGIF_from_3D



def read_and_display_nifiti_file(file_path, seg_path):
    # Load the .nii.gz file
    path_parts = file_path.split('/')
    output_filename = path_parts[-3] + '_' + path_parts[-2] + '_MIP.gif'
    img =  nib.load(file_path)
    pixdim = img.header['pixdim']
    data = img.get_fdata()
    seg_data = nib.load(seg_path).get_fdata()
    # print(img.header['pixdim'][1:4])
    # # create_mipGIF_from_3D
    # create_mipGIF_from_3D(img, nb_image = 2, is_mask=True)
    print("Start colored_seg_data")
    colored_seg_data = apply_color_coding_to_segmentation(seg_data, 0.6, pixdim[1:4])
    create_mipGIF_from_3D(data, pixdim, colored_seg_data,output_filename=output_filename, nb_image = 1, is_mask=False)
    # angle = 0
    # MIP_colored = project_colored_segmentation(colored_seg_data, angle)  # Assuming this function is defined elsewhere

    # print(img.header['pixdim'][1:4])
    # # Convert the image data to a numpy array
    # data = img.get_fdata()
    # print(f'data shape: {data.shape}')
    # # Display a specific slice (change the slice number as needed)
    # slice_number = 200 # Example slice number
    # plt.imshow(data[:, :, slice_number], cmap='gray')
    # plt.show()
    # return data


def main(file_path, seg_path):
    data = read_and_display_nifiti_file(file_path, seg_path)
    return data

if __name__ == "__main__":
    # file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_1285b86bea/02-24-2006-NA-PET-CT Ganzkoerper  primaer mit KM-49419/SUV.nii.gz'
    # seg_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_1285b86bea/02-24-2006-NA-PET-CT Ganzkoerper  primaer mit KM-49419/SEG.nii.gz'
    # file_path1 = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_1285b86bea/02-24-2006-NA-PET-CT Ganzkoerper  primaer mit KM-49419/SUV_MIP_output/SUV_MIP.nii.gz'
    # file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SUV.nii.gz'
    # seg_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SEG.nii.gz'
    # file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_5d553bf6b4/09-16-2001-NA-PET-CT Ganzkoerper  primaer mit KM-78907/SUV.nii.gz'
    # seg_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_5d553bf6b4/09-16-2001-NA-PET-CT Ganzkoerper  primaer mit KM-78907/SEG.nii.gz'
    file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_5d10be5b89/05-30-2005-NA-PET-CT Ganzkoerper  primaer mit KM-53829/SUV.nii.gz'
    seg_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_5d10be5b89/05-30-2005-NA-PET-CT Ganzkoerper  primaer mit KM-53829/SEG.nii.gz'
    main(file_path, seg_path)
    # main(file_path1)
 # Load the .nii.gz file
    # img =  nib.load(file_path)
    # print(img.header['pixdim'][1:4])
    # # Convert the image data to a numpy array
    # data = img.get_fdata()
    # print(f'data shape: {data.shape}')
    # # Display a specific slice (change the slice number as needed)
    # slice_number = 0 # Example slice number
    # plt.imshow(data[:, :, slice_number], cmap='gray')

# Load the .nii.gz file
# file_path0 = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_5d553bf6b4/09-16-2001-NA-PET-CT Ganzkoerper  primaer mit KM-78907/CTres.nii.gz'
# file_path0 = '/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SEG.nii.gz'
# file_path2 = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/tumor_1_seg.nii.gz'
# img =  nib.load(file_path0)
# seg_data = img.get_fdata()
# img = nib.load(file_path)q
# img2 = nib.load(file_path2)

# # # Convert the image data to a numpy array
# data = img.get_fdata()
# data2 = img2.get_fdata()
# print(np.all(data==data))
