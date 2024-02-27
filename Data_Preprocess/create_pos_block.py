from utils import get_connected_components_3D

import numpy as np
from utils import *
from tqdm import tqdm
def get_max_min_xyz_of_mask(binary_mask):
    # Find the non-zero indicesxs
    non_zero_indices = np.argwhere(binary_mask > 0)

    # Find min and max coordinates
    xmin, ymin, zmin = non_zero_indices.min(axis=0)
    xmax, ymax, zmax = non_zero_indices.max(axis=0)

    return {"x_min_coordinate": xmin, "y_min_coordinate": ymin, "z_min_coordinate": zmin, 
            "x_max_coordinate": xmax, "y_max_coordinate": ymax, "z_max_coordinate": zmax}
def get_tumor_under_threshold_block_coordinate(seg_mask, block_size = (3, 3, 3)):
    coordinates = get_max_min_xyz_of_mask(seg_mask)
    x_min, y_min, z_min = coordinates["x_min_coordinate"], coordinates["y_min_coordinate"], coordinates["z_min_coordinate"]
    x_max, y_max, z_max = coordinates["x_max_coordinate"], coordinates["y_max_coordinate"], coordinates["z_max_coordinate"]
    # Calculate the center coordinates
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    z_center = (z_min + z_max) // 2

    # Calculate half sizes for each dimension, adjusting for even sizes
    half_size_x = block_size[0] // 2
    half_size_y = block_size[1] // 2
    half_size_z = block_size[2] // 2

    x_start = max(x_center - half_size_x, 0)
    x_end = min(x_start + block_size[0], seg_mask.shape[0])
    y_start = max(y_center - half_size_y, 0)
    y_end = min(y_start + block_size[1], seg_mask.shape[1])
    z_start = max(z_center - half_size_z, 0)
    z_end = min(z_start + block_size[2], seg_mask.shape[2])

    # Correct the start positions if the block exceeds the mask dimensions
    if x_end - x_start < block_size[0]: x_start = x_end - block_size[0]
    if y_end - y_start < block_size[1]: y_start = y_end - block_size[1]
    if z_end - z_start < block_size[2]: z_start = z_end - block_size[2]
    return x_start, y_start, z_start
def get_tumor_under_threshold_block(coordinates, suv_data, block_size = (3, 3, 3)):
    x_start, y_start, z_start = coordinates
    x_end = x_start + block_size[0]
    y_end = y_start + block_size[1]
    z_end = z_start + block_size[2]
    block = suv_data[x_start:x_end, y_start:y_end, z_start:z_end]
    assert block.shape == block_size, f"Block shape is wrong and currently has shape of : {block.shape}"
    return block

def get_tumor_under_threshold_blocks_per_study(separate_seg_masks, suv_data, size_threshold = 3, block_size = (3, 3, 3)):
    block_lst = []
    coordinate_lst= []
    for mask in tqdm(separate_seg_masks):
        voxel = np.sum(mask)
        if voxel <= size_threshold:
            coordinates = get_tumor_under_threshold_block_coordinate(mask, block_size)
            block = get_tumor_under_threshold_block(coordinates, suv_data, block_size)
            block_lst.append(block)
            coordinate_lst.append(coordinates)
    return block_lst, coordinate_lst
    
    
