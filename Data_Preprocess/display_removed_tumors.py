from utils import *
from GIF_mip import *
from petct_utils import *
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import numpy as np
import imageio
import datetime
import glob
import shutil
from tqdm import tqdm
import scipy.ndimage
from matplotlib.colors import ListedColormap
import nibabel as nib

def apply_color_coding_to_segmentation(seg_data, threshold, pixdim):
    # Apply connected components to segment the tumors
    labels, num_features = cc3d.connected_components(seg_data, connectivity =26, return_N =True)
    
    # Placeholder for colored segmentation: Use a separate label (e.g., 2, 3) for different criteria
    colored_seg_data = np.zeros_like(seg_data)

    for i in range(1, num_features + 1):
        segment = labels == i
        voxel = np.sum(segment)
        # print(f"voxel: {voxel}")
        volume = voxel * np.prod(pixdim)/1000  # Simplistic size calculation; replace with actual size/volume calculation
        diameter = get_diameter_from_sphere_volume(volume)
        # Apply color coding based on size or other criteria
        if diameter < threshold:
            # print(f'Volume: {volume}, Diameter: {diameter}')
            colored_seg_data[segment] = 1  # Assign a specific color code for segments under the threshold
            # print(f"Segment {i} is under the threshold")
        else:
            colored_seg_data[segment] = 2  # Assign a different color code for segments meeting/exceeding the threshold
            # print(np.sum(colored_seg_data == 1))
    # print(np.unique(colored_seg_data))

    return colored_seg_data

def get_diameter_from_sphere_volume(volume):
    return 2 * (3 * volume / (4 * np.pi)) ** (1/3) 

def project_colored_segmentation(colored_seg_data, angle):
    # Rotate and project the colored segmentation
    vol_angle_colored = scipy.ndimage.rotate(colored_seg_data, angle, reshape=False, order=0)
    MIP_colored = np.amax(vol_angle_colored, axis=1)
    MIP_colored = np.flipud(MIP_colored.T)
    return MIP_colored

def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = f'Gif-{datetime.datetime.now().strftime("%Y-%M-%d-%H-%M-%S")}.gif'
    imageio.mimsave(output_file, images, duration=duration)
    print(f"GIF saved as {output_file}")

# def create_mipGIF_from_3D(img,nb_image=48,duration=0.2,is_mask=False,borne_max=None):
#     ls_mip=[]

#     img_data=img.get_fdata()
    
#     w = img.header['pixdim'][1] 
#     y = img.header['pixdim'][3] 
#     spacing = (1, y/w)
#     print('Pixel spacing ratio:', spacing)
    
#     liver_idx = img_data.shape[-1]//2
#     suv_liver = img_data[:,:,liver_idx].squeeze().max()
#     print('Liver SUV max', suv_liver)
    
#     print('Interpolating')
#     img_data+=1e-5
#     for angle in tqdm(np.linspace(0,360,nb_image)):
#         #ls_slice=[]
#         # This step is slow: https://stackoverflow.com/questions/14163211/rotation-in-python-with-ndimage
#         vol_angle= scipy.ndimage.interpolation.rotate(img_data,angle,order=0)
        
#         MIP=np.amax(vol_angle,axis=1)
#         MIP-=1e-5
#         MIP[MIP<1e-5]=0
#         MIP=np.flipud(MIP.T)
#         ls_mip.append(MIP)
#         print('angle:', angle)
    
#     try:
#         shutil.rmtree('MIP/')
#     except:
#         pass
#     os.mkdir('MIP/')
    
#     print('Creating MIP')
#     ls_image=[]
#     for mip,i in zip(ls_mip,range(len(ls_mip))):
#         fig,ax=plt.subplots()
#         ax.set_axis_off()
#         if borne_max is None:
#             if is_mask==True:
#                 borne_max=1
#             else:
#                 borne_max=suv_liver
#         plt.imshow(mip,cmap='Greys',vmax=borne_max)
#         # cmap = ListedColormap(['black', 'red', 'green', 'blue'])  # Example colormap for 4 segments
#         # plt.imshow(mip, cmap=cmap, vmax=borne_max, vmin=0)
#         fig.savefig('MIP/MIP'+'%04d' % (i)+'.png')
#         plt.close(fig)
#         interpolate_show_MIP(i, mip, borne_max, spacing=spacing)

#     filenames=glob.glob('MIP/*.png')


#     create_gif(filenames, duration)
# def create_mipGIF_from_3D(img_data,pixdim,nb_image=48,duration=0.2,is_mask=False,borne_max=None):
#     ls_mip=[]

#     # img_data=img.get_fdata()
#     w = pixdim[1]
#     y = pixdim[3]
    
#     # w = img.header['pixdim'][1] 
#     # y = img.header['pixdim'][3] 
#     spacing = (1, y/w)
#     print('Pixel spacing ratio:', spacing)
    
#     liver_idx = img_data.shape[-1]//2
#     suv_liver = img_data[:,:,liver_idx].squeeze().max()
#     print('Liver SUV max', suv_liver)
    
#     print('Interpolating')
#     img_data+=1e-5
#     for angle in tqdm(np.linspace(0,360,nb_image)):
#         #ls_slice=[]
#         # This step is slow: https://stackoverflow.com/questions/14163211/rotation-in-python-with-ndimage
#         vol_angle= scipy.ndimage.interpolation.rotate(img_data,angle,order=0)
        
#         MIP=np.amax(vol_angle,axis=1)
#         MIP-=1e-5
#         MIP[MIP<1e-5]=0
#         MIP=np.flipud(MIP.T)
#         print(MIP)
#         # print(f'MIP.shape', MIP.shape)
#         ls_mip.append(MIP)
#         print('angle:', angle)
    
#     try:
#         shutil.rmtree('MIP/')
#     except:
#         pass
#     os.mkdir('MIP/')
    
#     print('Creating MIP')
#     ls_image=[]
#     for mip,i in zip(ls_mip,range(len(ls_mip))):
#         fig,ax=plt.subplots()
#         ax.set_axis_off()
#         if borne_max is None:
#             if is_mask==True:
#                 borne_max=1
#             else:
#                 borne_max=suv_liver
#         plt.imshow(mip,cmap='Greys',vmax=borne_max)
#         # cmap = ListedColormap(['black', 'red', 'green', 'blue'])  # Example colormap for 4 segments
#         # plt.imshow(mip, cmap=cmap, vmax=borne_max, vmin=0)
#         fig.savefig('MIP/MIP'+'%04d' % (i)+'.png')
#         plt.close(fig)
#         interpolate_show_MIP(i, mip, borne_max, spacing=spacing)

#     filenames=glob.glob('MIP/*.png')


#     create_gif(filenames, duration)


def plot_each_study_tumor_removed(data_directory, patient_id, study_id, threshold = 0.6):
    suv_file_path = os.path.join(data_directory, patient_id, study_id, 'SUV.nii.gz')
    seg_file_path = os.path.join(data_directory, patient_id, study_id, 'SEG.nii.gz')
    output_filename = os.path.join(patient_id, study_id, 'MIP.gif') 
    suv_data = nib.load(suv_file_path).get_fdata()
    seg_data = nib.load(seg_file_path).get_fdata()
    pixdim = nib.load(seg_file_path).header['pixdim']
    print("Start colored_seg_data")
    colored_seg_data = apply_color_coding_to_segmentation(seg_data, threshold,pixdim[1:4])
    create_mipGIF_from_3D(suv_data, pixdim, colored_seg_data,output_filename=output_filename, nb_image = 1, is_mask=False)

def main(data_directory, patient_id, study_id):
    plot_each_study_tumor_removed(data_directory, patient_id, study_id)


if __name__ == "__main__":
    print('start!')
    data_directory = "/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions"
    # output_csv = 'all_patients_final_results_with_26_new_connected_components.csv'
    study_tumor_reomoved_file =  'Data_Preprocess/study tumor being filtered.npy'
    study_tumor_reomoved = np.load(study_tumor_reomoved_file, allow_pickle=True)
    for study in tqdm(study_tumor_reomoved):
        print(f"working with {study}")
        patient_id, study_id = study
        main(data_directory, patient_id, study_id)
        # break
    # main(data_directory, output_csv)
    print('done!')
    # pet_img = nib.load('/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/PET.nii.gz')
    # seg_img = nib.load('/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SEG.nii.gz')
    # create_mipGIF_from_3D(pet_img, seg_img, threshold=0.6, nb_image = 1)


