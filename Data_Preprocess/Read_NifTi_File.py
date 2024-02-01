import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
# Load the .nii.gz file
# file_path0 = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/CTres.nii.gz'
file_path0 = '/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SEG.nii.gz'
# file_path2 = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/tumor_1_seg.nii.gz'
img =  nib.load(file_path0)
seg_data = img.get_fdata()
# img = nib.load(file_path)q
# img2 = nib.load(file_path2)

# # # Convert the image data to a numpy array
# data = img.get_fdata()
# data2 = img2.get_fdata()
# print(np.all(data==data))


# Display a specific slice (change the slice number as needed)
# slice_number = 0 # Example slice number
# plt.imshow(data[:, :, slice_number], cmap='gray')
# plt.imshow(data[:, :, slice_number], cmap='gray', vmin=0, vmax=1)
print(img.header)
# # is the dimention in mm? The xyzt_units is 0, which means unknown?
# print(img.header['pixdim'][1:4].shape)
# print(np.all(get_connected_components_3D(seg_data)[1] == data2))
# # print(img.header)
# # print(np.all(data[:, :, slice_number]==0))
# plt.show()
