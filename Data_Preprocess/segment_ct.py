import os
import sys
import glob
import pathlib as plb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import math
import cv2
from scipy import ndimage
from utils import get_connected_components_3D
from pathlib import Path


# Windows a CT volume or slice
def window_ct_image(image, window_center, window_width, binarize=False):

    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    window_image = image.copy()
    window_image = np.clip(image, img_min, img_max)
    # print("------------ binarize the windowed image ------------")
    # plt.figure()
    # plt.imshow(window_image, cmap="gray", origin="lower")
    # plt.show()
    #binarize the windowed image
    if binarize:
        window_image[window_image != img_max] = 1
        window_image[window_image == img_max] = 0
        print("------------ binarize the windowed image ------------")
        plt.figure()
        plt.imshow(window_image, cmap="gray", origin="lower")
        plt.show()
        # print(np.all(window_image))
        # window_image = np.clip(image, -img_min, -img_max)
        # print(window_image)

        # window_image[window_image < img_min] = img_max
        # window_image[window_image > img_max] = img_min
    print(f'window_image shape after windowing: {window_image.shape}')

    return window_image, img_min, img_max
def apply_window(loaded_image, window_center, window_width, binarize=False):
    # image = nib.load(image_path).get_fdata()
    image = loaded_image.get_fdata()
    print(f'raw images shape: {image.shape}')
    window_image, img_min, img_max = window_ct_image(image, window_center, window_width, binarize)
    return window_image


def load_axial_image_slice(loaded_image, idx):
    # read pre-saved nifti images
    print('Loading images')
    images =loaded_image.get_fdata()
    print(f'raw images shape: {images.shape}')
    img = np.rot90(images[:,:,idx]).squeeze().copy()
    # img = np.rot90(images[:,:,:]).squeeze().copy()
    print(f'img shape after rot 90: {img.shape}')
    return img

# prep 1 axial CT image slice
# ct level and window can be read from the gaze jsons for each data point --?? need to check if saved properly
def axial_CT_slice(img, ct_level, ct_window, binarize):
    # img, img_min, img_max = window_ct_image(image, window_center, window_width)
    cmap_ct = plt.cm.gist_gray
    img, img_min, img_max = window_ct_image(img, ct_level, ct_window, binarize=binarize)
    norm_ct = plt.Normalize(vmin=img_min, vmax=img_max)
    img = cmap_ct(norm_ct(img))
    print(f'img shape after axial_ct_slice: {img.shape}')
    return img

def read_ct_slice(loaded_img, idx, dim = (400,400), ct_level = 50, ct_window = 100, binarize = False):
    print('loading image')
    img = load_axial_image_slice(loaded_img, idx)
    print('loaded image')
    # default shows soft tissue window
    img = axial_CT_slice(img, ct_level, ct_window, binarize)
    return img
    # if output_path:
    #     img_bgr = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR
    #     cv2.imwrite(output_path, img_bgr)
    # return img

def display_and_show_ct_slice(img, dim =(400,400), display = True, output_path = None):
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.namedWindow("slice", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("slice", dim[0],dim[1])
    while display:
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow('slice', img)
        if key == ord('q'):
            display = False
    cv2.destroyAllWindows()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create the directory if it does not exist
        print(output_path)
        gry = (img<1.1)
        img[gry]*=255
        cv2.imwrite(str(output_path),img)
def save_windowed_ct_by_tissue_type(path, tissue_type_lst):
    # Load the original NIfTI file to get the affine matrix
    loaded_raw_image = nib.load(path)
    original_affine = loaded_raw_image.affine

    for tissue_type in tissue_type_lst:
        binarize = False
        if tissue_type == "Lung" or tissue_type == "Fat":
            binarize = True
        print(f'working with {tissue_type}')
        ct_level = tissue_type_lst[tissue_type]["ct_level"]
        ct_window = tissue_type_lst[tissue_type]["ct_window"]
        # Load the original NIfTI file to get the affine matrix
        # applying window and save the image file
        windowed_image = apply_window(loaded_raw_image, ct_level, ct_window, binarize=binarize)
        # Create a Nifti1Image object with the windowed image data and the original affine matrix
        windowed_nifti_image = nib.Nifti1Image(windowed_image, affine=original_affine)
        # Save the Nifti1Image object to a new NIfTI file
        nifity_path = path.parent /f'{tissue_type}/windowed_ct_for_{tissue_type}.nii'
        nib.save(windowed_nifti_image, nifity_path)
    print('finish saving windowed ct images by tissue type')

def plot_ct_slice_by_tissue_type(path, tissue_type_lst, tissue_type, idx,display = True, output_path = False):
    # Load the original NIfTI file to get the affine matrix
    loaded_raw_image = nib.load(path)
    print(f'working with {tissue_type}')
    binarize = False
    if tissue_type == "Lung" or tissue_type == "Fat":
        binarize = True
    ct_level = tissue_type_lst[tissue_type]["ct_level"]
    ct_window = tissue_type_lst[tissue_type]["ct_window"]
    img = read_ct_slice(loaded_raw_image, idx, ct_level= ct_level, ct_window = ct_window, binarize=binarize)
    # print(f'img shape: {img.shape}')
    output_path = path.parent / f'{tissue_type}/ct_slice_visual/idx_{idx}.png'
    if output_path:
        display_and_show_ct_slice(img, output_path = output_path, display=True)
    else:
        display_and_show_ct_slice(img, output_path = None, display=True)
    # print(f'img shape: {img.shape}')
    # print(img)

    

if __name__ == "__main__":
    ct_file_path =  Path('/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/CTres.nii.gz')
    tissue_type = "Lung"
    tissue_type_lst ={
        "Bone":{"ct_level": 400, "ct_window": 100},
        "Soft Tissue": {"ct_level": 50, "ct_window": 100},
        "Fat": {"ct_level": -50, "ct_window": 50},
        "Lung": {"ct_level": -600, "ct_window": 200}
    }
    # for tissue_type in tissue_type_lst:
    # plot_ct_slice_by_tissue_type(ct_file_path, tissue_type_lst, tissue_type, 200)
    images = nib.load(ct_file_path).get_fdata()
    print(f'raw images shape: {images.shape}')
    img = np.rot90(images[:,:,200]).squeeze().copy()
    window_ct_image(img, -50, 50, binarize=True)
    # save_windowed_ct_by_tissue_type(ct_file_path, tissue_type_lst)
    
    # ct_level = tissue_type_lst[tissue_type]["ct_level"]
    # ct_window = tissue_type_lst[tissue_type]["ct_window"]
    # import nibabel as nib
    # # Load the original NIfTI file to get the affine matrix
    # loaded_raw_image = nib.load(ct_file_path)
    # original_affine = loaded_raw_image.affine

    # # applying window and save the image file
    # windowed_image = apply_window(loaded_raw_image, ct_level, ct_window)

    # # Create a Nifti1Image object with the windowed image data and the original affine matrix
    # windowed_nifti_image = nib.Nifti1Image(windowed_image, affine=original_affine)
    # # Save the Nifti1Image object to a new NIfTI file
    # nifity_path = ct_file_path.parent /f'{tissue_type}/windowed_ct_for_{tissue_type}.nii'
    # nib.save(windowed_nifti_image, nifity_path)
    # # plotting the windowed slice
    # idx = 200
    # img = read_ct_slice(loaded_raw_image, idx, ct_level= ct_level, ct_window = ct_window)
    # # print(f'img shape: {img.shape}')
    # output_path = ct_file_path.parent / f'{tissue_type}/ct_slice_visual/idx_{idx}.png'
    # display_and_show_ct_slice(img, output_path = output_path, display=False)
    # print(f'img shape: {img.shape}')
    # # print(img)
