import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the .nii.gz file
file_path = '/Users/wenyuanchen/Desktop/IBM/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SEG.nii.gz'
img = nib.load(file_path)

# Convert the image data to a numpy array
data = img.get_fdata()


# Display a specific slice (change the slice number as needed)
slice_number = 0 # Example slice number
# plt.imshow(data[:, :, slice_number], cmap='gray')
plt.imshow(data[:, :, slice_number], cmap='gray', vmin=0, vmax=1)
print(img.header)
# is the dimention in mm? The xyzt_units is 0, which means unknown?
# print(img.header['pixdim'][1:4].shape)
# print(np.all(data[:, :, slice_number]==0))
plt.show()
