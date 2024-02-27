### A COPIER POUR IMPORTER LES FONCTIONS
#import sys
#sys.path
#sys.path.append('/Users/Paul/Documents/soft/Python/my_func/')
#import my_functions_python
#
#
#

import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import nilearn
import shutil
import glob
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
import imageio
import datetime
import numpy as np
import scipy
from scipy import ndimage
from tqdm import tqdm
import pathlib as plb
import sys
import cv2
import pandas as pd
import display_removed_tumors as drt
import matplotlib.colors as mcolors
import pickle
import find_tumor_candidate


#### Create MIP GIF
'''def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)
    '''
def create_gif(filenames, duration, output_file):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = output_file + 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    print(output_file)
    imageio.mimsave(output_file, images, duration=duration)

    
## Interpolation to account for difference between pixel spacing and slice thickness 
# interpolation options: 'antialiased', 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'
# 8mm/2.03642 spacing = 3.92846269434 pixels/mm =~ 100dpi
# colormap options: https://matplotlib.org/stable/tutorials/colors/colormaps.html
def interpolate_show_MIP(i, nda, suv_max, spacing=(1,1), title=None, margin=0, dpi=100, colormap='Greys', OUTPATH=None,show=False):
    ysize = nda.shape[0]
    xsize = nda.shape[1]

    figsize = (1 + margin) * xsize * spacing[0] / dpi, (1 + margin) * ysize * spacing[1] / dpi

    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
    #hide axis
    ax.axis('off')
    
    extent = (0, xsize * spacing[0], 0, ysize * spacing[1])

    #various papers mentions bicubic interpolation...
    t = ax.imshow(
#         nda, extent=extent, interpolation="hamming", cmap="Greys", origin="upper", vmax=suv_max
#         nda, extent=extent, interpolation="bilinear", cmap="Greys", origin="upper", vmax=suv_max
        nda, extent=extent, interpolation="bicubic", cmap=colormap, origin="upper", vmax=suv_max 
    )

    if title:
        plt.title(title)
    if OUTPATH != None:
        fig.savefig(os.path.join(OUTPATH,'MIP'+'%04d' % (i)+'.png'), dpi = dpi)
    if not show:
        plt.close(fig)

    '''
def create_mipGIF_from_3D(img,nb_image=1,duration=0.2,is_mask=False,borne_max=None):
    ls_mip=[]

    img_data=img.get_fdata()
    
    w = img.header['pixdim'][1] 
    y = img.header['pixdim'][3] 
    spacing = (1, y/w)
    print('Pixel spacing ratio:', spacing)
    
    liver_idx = img_data.shape[-1]//2
    suv_liver = img_data[:,:,liver_idx].squeeze().max()
    print('Liver SUV max', suv_liver)
    
    print('Interpolating')
    img_data+=1e-5
    for angle in tqdm(np.linspace(0,360,nb_image)):
        #ls_slice=[]
        # This step is slow: https://stackoverflow.com/questions/14163211/rotation-in-python-with-ndimage
        vol_angle= scipy.ndimage.interpolation.rotate(img_data,angle,order=0)
        
        MIP=np.amax(vol_angle,axis=1)
        MIP-=1e-5
        MIP[MIP<1e-5]=0
        MIP=np.flipud(MIP.T)
        print(MIP)
        # print(f'MIP.shape', MIP.shape)
        ls_mip.append(MIP)
        print('angle:', angle)
    
    try:
        shutil.rmtree('MIP/')
    except:
        pass
    os.mkdir('MIP/')
    
    print('Creating MIP')
    ls_image=[]
    for mip,i in zip(ls_mip,range(len(ls_mip))):
        fig,ax=plt.subplots()
        ax.set_axis_off()
        if borne_max is None:
            if is_mask==True:
                borne_max=1
            else:
                borne_max=suv_liver
        plt.imshow(mip,cmap='Greys',vmax=borne_max)
        # cmap = ListedColormap(['black', 'red', 'green', 'blue'])  # Example colormap for 4 segments
        # plt.imshow(mip, cmap=cmap, vmax=borne_max, vmin=0)
        fig.savefig('MIP/MIP'+'%04d' % (i)+'.png')
        plt.close(fig)
        interpolate_show_MIP(i, mip, borne_max, spacing=spacing)

    filenames=glob.glob('MIP/*.png')


    create_gif(filenames, duration)
    '''
        
def create_mipGIF_from_3D(img_data, pixdim, colored_seg_data, output_filename, nb_image=2, duration=0.2,is_mask=False,borne_max=None):
    output_dir = os.path.join('positive patch candidate', output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ls_mip=[]
    ls_seg_mip = []

    # img_data=img.get_fdata()
    w = pixdim[1]
    y = pixdim[3]
    
    # w = img.header['pixdim'][1] 
    # y = img.header['pixdim'][3] 
    spacing = (1, y/w)
    print('Pixel spacing ratio:', spacing)
    
    liver_idx = img_data.shape[-1]//2
    suv_liver = img_data[:,:,liver_idx].squeeze().max()
    print('Liver SUV max', suv_liver)
    
    print('Interpolating')
    img_data+=1e-5
    for angle in tqdm(np.linspace(0,360,nb_image)):
        #ls_slice=[]
        # This step is slow: https://stackoverflow.com/questions/14163211/rotation-in-python-with-ndimage
        vol_angle= scipy.ndimage.interpolation.rotate(img_data,angle,order=0)
        
        MIP=np.amax(vol_angle,axis=1)
        MIP-=1e-5
        MIP[MIP<1e-5]=0
        MIP=np.flipud(MIP.T)
        # print(np.unique(MIP))
        # print(f'MIP.shape', MIP.shape)
        ls_mip.append(MIP)
        ls_seg_mip.append(drt.project_colored_segmentation(colored_seg_data, angle))
        print('angle:', angle)
    
    try:
        shutil.rmtree('MIP/')
    except:
        pass
    # os.mkdir('MIP/')
    
    print('Creating MIP')
    ls_image=[]
    for mip,seg_mip,i in zip(ls_mip,ls_seg_mip,range(len(ls_mip))):
        fig,ax=plt.subplots()
        ax.set_axis_off()
        if borne_max is None:
            if is_mask==True:
                # borne_max=1
                borne_max = 1
            else:
                borne_max=suv_liver
        plt.imshow(mip,cmap='Greys',vmax=borne_max)

        # print(seg_mip[150])
        # maksed value for 2 and keep value for 1 and thus plotting value for 1
        # mask_for_1 = np.array(seg_mip == 2)
        mask_for_1 = np.array((seg_mip == 2) | (seg_mip == 3))
        # print(f"mask_for_1{mask_for_1[150]}")
        # print("sum of mask_for_1")
        # print(np.sum(mask_for_1))

        # maksed value for 1 and keep value for 2 and thus plotting value for 2
        # mask_for_2 = np.array(seg_mip == 1)
        mask_for_2 = np.array(seg_mip == 1 | (seg_mip == 3))
        mask_for_3 = np.array(seg_mip == 2 | (seg_mip == 1))

        # print("sum of mask_for_2")
        # print(np.sum(mask_for_2))

        # print(np.unique(seg_mip))
        print("------start overlaying------")
        # Overlay green for pixels corresponding to value 1 in seg_mip
        # print("---------")
        # print(np.ma.masked_where(mask_for_1, seg_mip))
        mask_1 = np.ma.masked_where(mask_for_1, seg_mip)
        # print(f"mask_1: {mask_1[150]}")
        mask_2 = np.ma.masked_where(mask_for_2, seg_mip)
        mask_3 = np.ma.masked_where(mask_for_3, seg_mip)
        plt.imshow(mask_1, cmap=mcolors.ListedColormap(['none','red']), interpolation='none', alpha=0.5)

        # Overlay red for pixels corresponding to value 2 in seg_mip
        # print("sum for mask_for_1_plot")
        # print(np.sum(mask_1))

        plt.imshow(mask_2, cmap=mcolors.ListedColormap(['none', 'green']), interpolation='none', alpha=0.5)
        plt.imshow(mask_3, cmap=mcolors.ListedColormap(['none', 'blue']), interpolation='none', alpha=0.5)
        # print("sum for mask_for_2_plot")
        # print(np.sum(mask_2))

        # plt.show()
 
        # plt.imshow(mip, cmap=cmap, vmax=borne_max, vmin=0)
        output_filepath = os.path.join(output_dir, 'MIP' + '%04d' % (i) + '.png')
        fig.savefig(output_filepath)
        plt.close(fig)
        # interpolate_show_MIP(i, mip, borne_max, spacing=spacing, OUTPATH=output_dir, show=False)

        # fig.savefig('MIP/MIP'+'%04d' % (i)+'.png')
        # plt.close(fig)
        interpolate_show_MIP(i, mip, borne_max, spacing=spacing)

    # filenames=glob.glob('MIP/*.png')


    # create_gif(filenames, duration)
    filenames = glob.glob(os.path.join(output_dir, '*.png'))
    # output_gif_path = os.path.join(output_dir, 'animation.gif')
    # create_gif(filenames, duration, output_gif_path)

    
#     try:
#         shutil.rmtree('test_gif/')
#     except:
#         pass

# Pad to same shape before stack
def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    x_pad = abs((x_-x))
    y_pad = abs((y_-y)) # should be 0
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant') #Default is 0


# Makes more sense to return/save the interpolated MIP as nifti so can read original SUV pixels back, which allows SUV adjustment later
def create_mipNIFTI_from_3D(img, nb_image=48):
    
    ls_mip=[]
    
    img_data=img.get_fdata()
    shape = img.get_fdata().shape
    max_dim = max(shape)
    diag = int(np.ceil(np.sqrt(np.square(shape[0])+np.square(shape[1]))))
#     max_dim = max(max_dim, diag)
#     target_shape = (max_dim,max_dim)
    # adapted this so doesn't need to crop/reschale again.
    max_width = max([450, diag])
    target_shape = (shape[2], max_width) # (y_, x_)
    print(target_shape)
    
    #Modified nifti header saving useful axial slices information
    # Can't seem to create new fields but can use existing fields to store other information...
    header = img.header.copy()
    liver_idx = img_data.shape[-1]//2
    suv_liver = img_data[:,:,liver_idx].squeeze().max()
    suv_brain = img_data[:,:,-1].squeeze().max()
#     print('Liver SUV max', suv_liver)
    header['intent_p1'] = suv_liver
    header['intent_p2'] = suv_brain
    header['intent_p3'] = img_data.max()
    # Can't store too many letters...
    header['intent_name'] = b'liver;brain;max'
    
#     print('Interpolating')
    img_data+=1e-5
    for angle in tqdm(np.linspace(0,360,nb_image)):
        #ls_slice=[]
        # This step is slow: https://stackoverflow.com/questions/14163211/rotation-in-python-with-ndimage
#         vol_angle= scipy.ndimage.interpolation.rotate(img_data,angle,order=0)
        vol_angle = scipy.ndimage.rotate(img_data,angle,order=0)
        
        MIP=np.amax(vol_angle,axis=1)
        MIP-=1e-5
        MIP[MIP<1e-5]=0
        MIP=np.flipud(MIP.T)
        MIP=to_shape(MIP, target_shape)
        ls_mip.append(MIP)
#         print('angle:', angle, MIP.shape)
    
    new_data = np.dstack(ls_mip) #shape [:,:,i]
    mip_nifti = nib.Nifti1Image(new_data, None, header)
    
    return mip_nifti


def rescale_mipNIFTI(img, seg):
    ls_mip=[]
    ls_seg=[]
    
    img_data = img.get_fdata()
    header = img.header.copy()
    nb_image = img_data.shape[2]
    seg_data = seg.get_fdata()
    seg_header = seg.header.copy()

    max_edge = 0
    for idx in range(nb_image):
        mip = img_data[:,:,idx].copy()
        seg_mip = seg_data[:,:,idx].copy()
        
        coords = cv2.findNonZero(mip)
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        target_shape = (w, int(header['pixdim'][3]/header['pixdim'][1]*h))
        max_edge = max([max_edge, np.max(target_shape)])
        
        rect = mip[y:y+h, x:x+w]
        seg_rect = seg_mip[y:y+h, x:x+w]
        
        new_slice = cv2.resize(rect, dsize=target_shape, interpolation=cv2.INTER_LINEAR)
        ls_mip.append(new_slice)
        new_seg = cv2.resize(seg_rect, dsize=target_shape, interpolation=cv2.INTER_LINEAR)
        ls_seg.append(new_seg)

    ls_padded = []
    ls_seg_padded = []
    for new_slice, new_seg in zip(ls_mip, ls_seg):
        new_padded = to_shape(new_slice, (max_edge,max_edge))
        ls_padded.append(new_padded)
        seg_padded = to_shape(new_seg, (max_edge,max_edge))
        ls_seg_padded.append(seg_padded)
    
    new_data = np.dstack(ls_padded) #shape [:,:,i]
    mip_nifti = nib.Nifti1Image(new_data, None, header)
    new_seg_data = np.dstack(ls_seg_padded) #shape [:,:,i]
    seg_nifti = nib.Nifti1Image(new_seg_data, None, seg_header)
    
    return mip_nifti, seg_nifti


def find_studies(path_to_data):
    # find all studies
    root = plb.Path(path_to_data)
    patient_dirs = list(root.glob('*'))

    study_dirs = []

    for dir in patient_dirs:
        sub_dirs = list(dir.glob('*'))
        #print(sub_dirs)
        study_dirs.extend(sub_dirs)

    return study_dirs


def find_unprocessed_studies(path_to_data, nii_out_root):
    study_dirs = find_studies(path_to_data)
    #print(len(study_dirs))
    study_out_dirs = find_studies(nii_out_root)
    #print(len(study_out_dirs))

    processed_pts = []
    for study_out_dir in study_out_dirs:
        # study_out_dir.parent.name not unique enough
        patient = study_out_dir.name
        processed_pts.append(patient)
    #print(len(set(processed_pts)))

    unprocessed_study_dirs = []
    for study_dir in study_dirs:
        patient = study_dir.name
        if patient in processed_pts:
            continue
    #         print("The following patient directory has been processed: ", patient)
        else: 
            unprocessed_study_dirs.append(study_dir)
    #         print("The following patient directory is being processed: ", patient)
    # print(len(unprocessed_study_dirs))
    
    return unprocessed_study_dirs


def convert_axial_niis_to_MIP(study_dir, nii_out_root):
    # batch conversion of all patients
    # for study_dir in tqdm(study_dirs):
    print("The following patient directory is being processed: ", study_dir)
    
    # patient = study_dir.parent.name
    patient = study_dir
    print("The following patient directory is being processed: ", patient)
    
    # Preserving same diretory structure as original tcia dataset
    # nii_out_path = plb.Path(nii_out_root/study_dir.parent.name)
    # nii_out_path = nii_out_path/study_dir.name
    nii_out_path = nii_out_root
    os.makedirs(nii_out_path, exist_ok=True) #leaves dir unaltered if already exists
    
    print('Processing SUV.nii.gz', patient)
    img = nib.load(os.path.join(study_dir, 'SUV.nii.gz'))
    mip_nifti = create_mipNIFTI_from_3D(img, nb_image=48) #48 is the number of MIP slices in MIM available to rads
    nib.save(mip_nifti, os.path.join(nii_out_path, 'SUV_MIP.nii.gz'))
    
    print('Processing SEG.nii.gz', patient)
    img = nib.load(os.path.join(study_dir, 'SEG.nii.gz'))
    mip_nifti = create_mipNIFTI_from_3D(img, nb_image=48) #48 is the number of MIP slices in MIM available to rads
    nib.save(mip_nifti, os.path.join(nii_out_path, 'SEG_MIP.nii.gz'))

        
def rescale_all_MIP(study_dirs, nii_out_root):
    # batch rescaling of all patients
    for study_dir in tqdm(study_dirs):
        
        patient = study_dir.parent.name
        print("The following patient directory is being processed: ", patient)
        
        # Preserving same diretory structure as original tcia dataset
        nii_out_path = plb.Path(nii_out_root/study_dir.parent.name)
        nii_out_path = nii_out_path/study_dir.name
        os.makedirs(nii_out_path, exist_ok=True) #leaves dir unaltered if already exists
        
        print('Processing SUV_MIP.nii.gz and SEG_MIP.nii.gz for:', patient)
        img = nib.load(os.path.join(study_dir, 'SUV_MIP.nii.gz'))
        seg = nib.load(os.path.join(study_dir, 'SEG_MIP.nii.gz'))
        mip_rescale, seg_rescale = rescale_mipNIFTI(img, seg)
        nib.save(mip_rescale, os.path.join(nii_out_path, 'SUV_MIP_rescale.nii.gz'))
        nib.save(seg_rescale, os.path.join(nii_out_path, 'SEG_MIP_rescale.nii.gz'))
    
    
# will index studies with original dicom nested directory paths
def process_nested_studies_dirs(nii_in_root, nii_out_root, unprocessed_only = False):
    if unprocessed_only:
        study_dirs = find_unprocessed_studies(nii_in_root, nii_out_root)
    else:
        study_dirs = find_studies(nii_in_root)    
    convert_axial_niis_to_MIP(study_dirs, nii_out_root)
    
    
# assumes a flat dir of nifty studies       
def process_flat_nifties_dir(nii_in_root, nii_out_root, study_IDs):
    os.makedirs(nii_out_root, exist_ok=True)
    
    # find all studies
    root = plb.Path(nii_in_root)
    full_nifty_paths = []
    for study_id in study_IDs:
        study_path = list(root.glob('{}_*deformedSeg.nii.gz'.format(study_id)))
        #study_path = list(root.glob('{}_*[Ss][Uu][Vv]*.nii.gz'.format(study_id)))
        #study_path = list(root.glob('{}_*[Ss][Ee][Gg]*.nii.gz'.format(study_id)))
        full_nifty_paths.extend(study_path)
        #print('###', len(full_nifty_paths))
    # axial nifty to MIP nifty for every .nii.gz file in the nii_in_root directory with filename containing study_type
    for nifty_path in tqdm(full_nifty_paths):
        # Preserving same diretory structure as original tcia dataset
        nii_out_path = plb.Path(nii_out_root/nifty_path.name)
        
        print('Processing nifty:', nii_out_path)
        img = nib.load(nifty_path)
        mip_nifti = create_mipNIFTI_from_3D(img, nb_image=48) #48 is the number of MIP slices in MIM available to rads
        nib.save(mip_nifti, nii_out_path)

        
# Process all the SUV.nii.gz to a MIP_SUV.nii.gz
if __name__ == "__main__":
    # filtered_separate_seg_masks_file = 'filtered_separate_seg_masks_no_pos.pkl'
    # if os.path.exists(filtered_separate_seg_masks_file):
    #     # Load the data from the file if it exists
    #     with open(filtered_separate_seg_masks_file, 'rb') as f:
    #         filtered_separate_seg_masks = pickle.load(f)
    from utils import get_connected_components_3D
    from create_pos_block import *
    suv_file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_f21755a99b/05-05-2005-NA-PET-CT Ganzkoerper  primaer mit KM-44651/SUV.nii.gz'
    seg_file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_f21755a99b/05-05-2005-NA-PET-CT Ganzkoerper  primaer mit KM-44651/SEG.nii.gz'
    suv_img = nib.load(suv_file_path)
    seg_img = nib.load(seg_file_path)
    suv_data = suv_img.get_fdata()
    seg_data = seg_img.get_fdata()
    separate_seg_masks = get_connected_components_3D(seg_data, connectivity=26)
    print(f"separate_seg_masks shape: {len(separate_seg_masks)}")
    boundary_seg = np.zeros(seg_data.shape)
    boundary_seg[seg_data == 1] = 2
    block_size = (3, 3, 3)
    size_threshold = 3
    for mask in separate_seg_masks:
        if np.sum(mask) <= size_threshold:
            boundary_seg[mask == 1] = 3

    block_lst, coordinate_lst = get_tumor_under_threshold_blocks_per_study(separate_seg_masks, suv_data, size_threshold = 3, block_size = block_size)
    for i in range(len(coordinate_lst)):
        # print(f"block_lst[{i}]: {block_lst[i].shape}")
        x_start, y_start, z_start = coordinate_lst[i]
        x_end = x_start + block_size[0]
        y_end = y_start + block_size[1]
        z_end = z_start + block_size[2]
        boundary_seg[x_start:x_end, y_start:y_end, z_start:z_end] = 1
        boundary_seg[x_start, y_start, z_start] = 1
        boundary_seg[x_end, y_end, z_end] = 1


    create_mipGIF_from_3D(suv_data, suv_img.header['pixdim'], boundary_seg, 'hongzhi', nb_image=1)

    
    '''
    # hongzhi's method
    import numpy as np
    from skimage.feature import peak_local_max
    maksed_out_suv = np.where(seg_data, 0, suv_data)
    R= 2
    TH=2.0
    local_max=peak_local_max(maksed_out_suv,min_distance=R,threshold_abs=TH)
    local_maximums = np.zeros(suv_data.shape)
    for i in range(len(local_max)):
        local_maximums[local_max[i][0],local_max[i][1],local_max[i][2]]=1
    # local_maximums[seg_data == 1] = 0'''
    
    

    '''# Assuming your 10 binary masks are named binary_mask1, binary_mask2, ..., binary_mask10
    # For this example, I'll create an array of masks for demonstration; replace this with your masks

    # Initialize the final segmentation map as a copy of the first mask to keep the same shape
    segmentation_map = np.copy(filtered_separate_seg_masks[0])

    # Perform a logical OR operation across all masks
    for mask in filtered_separate_seg_masks[1:]:  # Start from the second mask since we already copied the first one
        segmentation_map = np.logical_or(segmentation_map, mask)
        # break

    # Convert the logical array to an integer array if needed (1s and 0s)
    segmentation_map = segmentation_map.astype(int)
    segmentation_map[seg_data == 1] = 2'''

    # 'segmentation_map' now is a binary mask where 1 indicates the presence of any object



    # create_mipGIF_from_3D(suv_img)
    # create_mipGIF_from_3D(suv_data, suv_img.header['pixdim'], local_maximums, 'hongzhi', nb_image=1)
    # nii_in_root = plb.Path(sys.argv[1])  # path to parent directory for all studies, e.g. '...datasets/NIFTI/FDG-PET-CT-Lesions/'
    # nii_out_root = plb.Path(sys.argv[2])  # path to where we want to MIP nifti files, e.g. '...datasets/NIFTI_MIP/FDG-PET-CT-Lesions/')
    # ID_file = plb.Path(sys.argv[3]) # 1 for yes, 0 for no
    
    # if ID_file == '.':
    #     # uses nested file paths as ID
    #     process_nested_studies_dirs(nii_in_root, nii_out_root, unprocessed_only = False)
    # else:
    #     df = pd.read_csv(ID_file)
    #     study_IDs = df['ID'].tolist()
    #     process_flat_nifties_dir(nii_in_root, nii_out_root, study_IDs = study_IDs)
    
    # process_flat_nifties_dir
    # convert_axial_niis_to_MIP('/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_1285b86bea/02-24-2006-NA-PET-CT Ganzkoerper  primaer mit KM-49419',
                            #    '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_1285b86bea/02-24-2006-NA-PET-CT Ganzkoerper  primaer mit KM-49419/SUV_MIP_output')
    # converting axial to MIP (too much padding)
    #convert_axial_niis_to_MIP(study_dirs, nii_out_root)
    # Rescaling to remove white space border and to get same aspect ratio per PET affine 
    # Assumes have run prior axial to MIP conversion already. Can write out in same directory, e.g.:
    # python GIF_mip.py /gpfs/fs0/data/stanford_data/petct/NIFTI_MIP/FDG-PET-CT-Lesions/ /gpfs/fs0/data/stanford_data/petct/NIFTI_MIP/FDG-PET-CT-Lesions/
    # python GIF_mip.py /media/storage/Joy/datasets/NIFTI_MIP/FDG-PET-CT-Lesions/ /media/storage/Joy/datasets/NIFTI_MIP/FDG-PET-CT-Lesions/
    ### rescale_all_MIP(study_dirs, nii_out_root) # shouldn't need this anymore
    
    #done # python GIF_mip.py /gpfs/fs0/data/stanford_data/master/image_for_train_processed/ /gpfs/fs0/data/stanford_data/petmr_detr_dataset/baseline_test/SUV_MIP/ '/gpfs/fs0/data/stanford_data/petmr_detr_dataset/Baseline_Test_IDs.csv'
    #done # python GIF_mip.py /gpfs/fs0/data/stanford_data/master/annotation_for_train/ /gpfs/fs0/data/stanford_data/petmr_detr_dataset/baseline_test/SEG_MIP/ '/gpfs/fs0/data/stanford_data/petmr_detr_dataset/Baseline_Test_IDs.csv'
    
    #done #python GIF_mip.py /gpfs/fs0/data/stanford_data/master/image_for_train_processed/ /gpfs/fs0/data/stanford_data/petmr_detr_dataset/baseline_dev/SUV_MIP/ '/gpfs/fs0/data/stanford_data/petmr_detr_dataset/Baseline_Dev_IDs.csv'
    #done #python GIF_mip.py /gpfs/fs0/data/stanford_data/master/annotation_for_train/ /gpfs/fs0/data/stanford_data/petmr_detr_dataset/baseline_dev/SEG_MIP/ '/gpfs/fs0/data/stanford_data/petmr_detr_dataset/Baseline_Dev_IDs.csv'
    
    #done #python GIF_mip.py /gpfs/fs0/data/stanford_data/followup_shashi/ /gpfs/fs0/data/stanford_data/petmr_detr_dataset/followup/SUV_MIP/ '/gpfs/fs0/data/stanford_data/petmr_detr_dataset/FU_IDs.csv'
    #done #python GIF_mip.py /gpfs/fs0/data/stanford_data/followup_shashi/ /gpfs/fs0/data/stanford_data/petmr_detr_dataset/followup/SEG_MIP/ '/gpfs/fs0/data/stanford_data/petmr_detr_dataset/FU_IDs.csv'
    
