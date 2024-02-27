import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from utils import *
from tqdm import tqdm
import pickle

def filter_out_small_connected_components(separate_seg_masks, SUV_TH = 3.0, pixel_volume_TH = 10):
    filtered_separate_seg_masks = []
    for mask in tqdm(separate_seg_masks):
        if (np.max(mask) >= SUV_TH )or (np.sum(mask) >= pixel_volume_TH):
            filtered_separate_seg_masks.append(mask)
    return filtered_separate_seg_masks
def caulcate_IoU_box(gt_mask, pred_mask):
    gt_mask_box_coord = get_max_min_xyz_of_mask(gt_mask)
    pred_mask_box_coord = get_max_min_xyz_of_mask(pred_mask)
    x_overlap = max(0, min(gt_mask_box_coord["x_max_coordinate"], pred_mask_box_coord["x_max_coordinate"]) - max(gt_mask_box_coord["x_min_coordinate"], pred_mask_box_coord["x_min_coordinate"]))
    y_overlap = max(0, min(gt_mask_box_coord["y_max_coordinate"], pred_mask_box_coord["y_max_coordinate"]) - max(gt_mask_box_coord["y_min_coordinate"], pred_mask_box_coord["y_min_coordinate"]))
    z_overlap = max(0, min(gt_mask_box_coord["z_max_coordinate"], pred_mask_box_coord["z_max_coordinate"]) - max(gt_mask_box_coord["z_min_coordinate"], pred_mask_box_coord["z_min_coordinate"]))
    intersection = x_overlap * y_overlap * z_overlap
    if intersection == 0:
        return 0
    gt_mask_volume = (gt_mask_box_coord["x_max_coordinate"] - gt_mask_box_coord["x_min_coordinate"]) * (gt_mask_box_coord["y_max_coordinate"] - gt_mask_box_coord["y_min_coordinate"]) * (gt_mask_box_coord["z_max_coordinate"] - gt_mask_box_coord["z_min_coordinate"])
    pred_mask_volume = (pred_mask_box_coord["x_max_coordinate"] - pred_mask_box_coord["x_min_coordinate"]) * (pred_mask_box_coord["y_max_coordinate"] - pred_mask_box_coord["y_min_coordinate"]) * (pred_mask_box_coord["z_max_coordinate"] - pred_mask_box_coord["z_min_coordinate"])
    union = gt_mask_volume + pred_mask_volume - intersection
    iou = intersection / union
    return iou

def calcualte_IoU_mask(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    if intersection.sum() == 0:
        return 0
    union = np.logical_or(gt_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def find_best_matched_gt_mask_and_iou(filtered_separate_seg_masks, ground_truth_seg_masks,boundary_box = False, both = False):
    assert not ((boundary_box == False) and (both == True)), "both cannot be True if boundary_box isn't true at the same time"
    if both and boundary_box:
        best_matched_gt_mask_mask = []
        best_matched_gt_mask_box = []
        for candidate_idx, mask in  enumerate(tqdm(filtered_separate_seg_masks)):
            iou_scores_box = []
            iou_scores_mask = []
            for idx, gt_mask in enumerate(tqdm(ground_truth_seg_masks)):
                box_score = caulcate_IoU_box(gt_mask, mask)
                mask_score = calcualte_IoU_mask(gt_mask, mask)
                iou_scores_box.append(box_score)
                iou_scores_mask.append(mask_score)
            max_iou_box = np.max(iou_scores_box)
            max_iou_mask = np.max(iou_scores_mask)
            if max_iou_box > 0:
                print(f"iou score: {max_iou_box}")
                best_matched_gt_mask_box.append({"candidate_patch_idx": candidate_idx, "best_matched_gt_mask_idx": np.argmax(iou_scores_mask), "iou_score": max_iou_mask})
            # best_matched_gt_mask.append(ground_truth_seg_masks[np.argmax(iou_scores)])
            if max_iou_mask > 0:
                best_matched_gt_mask_mask.append({"candidate_patch_idx": candidate_idx,"best_matched_gt_mask_idx": np.argmax(iou_scores_box), "iou_score": max_iou_box})
        return best_matched_gt_mask_mask, best_matched_gt_mask_box

    else:
        best_matched_gt_mask = []
        for candidate_idx, mask in enumerate(tqdm(filtered_separate_seg_masks)):
            iou_scores = []
            for idx, gt_mask in enumerate(tqdm(ground_truth_seg_masks)):
                if boundary_box:
                    score = caulcate_IoU_box(gt_mask, mask)
                    iou_scores.append(score)
                else:
                    score = calcualte_IoU_mask(gt_mask, mask)
                    # if score > 0:
                    #     print(f"iou score: {score}")
                    iou_scores.append(score)
            max_iou = np.max(iou_scores)
            if max_iou > 0:
                print(f"iou score: {max_iou}")
                best_matched_gt_mask.append({"candidate_patch_idx": candidate_idx, "best_matched_gt_mask_idx": np.argmax(iou_scores), "iou_score": np.max(iou_scores)})

            # best_matched_gt_mask.append({"best_matched_gt_mask_idx": ground_truth_seg_masks[np.argmax(iou_scores)], "iou_score": np.max(iou_scores)})
            # best_matched_gt_mask.append(ground_truth_seg_masks[np.argmax(iou_scores)])
        return best_matched_gt_mask

def get_max_min_xyz_of_mask(binary_mask):
    # Find the non-zero indicesxs
    non_zero_indices = np.argwhere(binary_mask > 0)

    # Find min and max coordinates
    xmin, ymin, zmin = non_zero_indices.min(axis=0)
    xmax, ymax, zmax = non_zero_indices.max(axis=0)

    return {"x_min_coordinate": xmin, "y_min_coordinate": ymin, "z_min_coordinate": zmin, 
            "x_max_coordinate": xmax, "y_max_coordinate": ymax, "z_max_coordinate": zmax}

if __name__ == "__main__":
    suv_file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SUV.nii.gz'
    seg_file_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/PETCT_0b57b247b6/05-02-2002-NA-PET-CT Ganzkoerper  primaer mit KM-42966/SEG.nii.gz'
    suv_img = nib.load(suv_file_path)
    seg_img = nib.load(seg_file_path)
    suv_data = suv_img.get_fdata()
    seg_data = seg_img.get_fdata()
    print(f"seg_data sum: {np.sum(seg_data)}")
    # using SUV TH = 2.0
    binary_suv_data_mask = np.zeros(seg_data.shape)
    binary_suv_data_mask[suv_data >= 2.0] = 1
    print(f"binary_suv_data_mask sum: {np.sum(binary_suv_data_mask)}")
    
    binary_suv_data_mask[seg_data == 1] = 0
    print(f"binary_suv_data_mask sum: {np.sum(binary_suv_data_mask)}")
    # get the connected components for the binary mask
    separate_candidate_seg_masks = get_connected_components_3D(binary_suv_data_mask, connectivity = 26)
    ground_truth_seg_masks = get_connected_components_3D(seg_data, connectivity = 26)
    print(f"separate_seg_masks shape: {len(separate_candidate_seg_masks)}") # 1054
    # filter out the small connected components either max SUV < 3.0 or pixel volume < 10
    filtered_separate_seg_masks = filter_out_small_connected_components(separate_candidate_seg_masks)
    ...
    

    # filtered_separate_seg_masks = None
    filtered_separate_seg_masks_file = 'filtered_separate_seg_masks_no_pos.pkl'
    # if os.path.exists(filtered_separate_seg_masks_file):
    #     # Load the data from the file if it exists
    #     with open(filtered_separate_seg_masks_file, 'rb') as f:
    #         filtered_separate_seg_masks = pickle.load(f)
    # else:
    #     # Calculate the filtered separate seg masks if the file doesn't exist
    #     filtered_separate_seg_masks = filter_out_small_connected_components(separate_candidate_seg_masks)
    #     # Save the filtered separate seg masks to a file
    with open(filtered_separate_seg_masks_file, 'wb') as f:
            pickle.dump(filtered_separate_seg_masks, f)

    # print(f"filtered_separate_seg_masks shape: {len(filtered_separate_seg_masks)}") # 252/253

    # ...
    print(f"filtered_separate_seg_masks shape: {len(filtered_separate_seg_masks)}") #252
    # find_best_matched_gt_mask_and_iou(filtered_separate_seg_masks, ground_truth_seg_masks, boundary_box = False, both = True) #assert error
    # mask_output = find_best_matched_gt_mask_and_iou(filtered_separate_seg_masks, ground_truth_seg_masks)
    # print(mask_output)
    # print(len(mask_output))



    # find the best matched ground truth mask for each candidate mask
