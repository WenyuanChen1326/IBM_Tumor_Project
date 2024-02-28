import numpy as np
# np.random.seed(0)  # Ensure reproducibility


def sample_neg_block(seg_data, local_max_points_coordinate_without_pos_coordinate, block_size=(3, 3, 3), sample_size=10, filtered_separate_segmentation_masks = None):
    max_attempts = 100
    # attempt = 0
    # block_found = False
    adjusted_smaple_size = min(sample_size, len(local_max_points_coordinate_without_pos_coordinate))
    neg_output_coord_list = []
    success_count = 0
    while success_count < adjusted_smaple_size:
        attempt = 0
        block_found = False
        while attempt < max_attempts and not block_found:
            # # Generate random starting indices
            # start_x = np.random.randint(0, seg_data.shape[0] - 2)  # -2 to include the end index for a 3x3x3 block
            # start_y = np.random.randint(0, seg_data.shape[1] - 2)
            # start_z = np.random.randint(0, seg_data.shape[2] - 2)
            # index = np.random.choice(len(local_max_points_coordinate_without_pos_coordinate))
            index = np.random.choice(range(len(local_max_points_coordinate_without_pos_coordinate)))
            x_center, y_center, z_center = local_max_points_coordinate_without_pos_coordinate[index]
            # Calculate half sizes for each dimension, adjusting for even sizes
            half_size_x = block_size[0] // 2
            half_size_y = block_size[1] // 2
            half_size_z = block_size[2] // 2
            x_start = max(x_center - half_size_x, 0)
            x_end = min(x_start + block_size[0], seg_data.shape[0])
            y_start = max(y_center - half_size_y, 0)
            y_end = min(y_start + block_size[1], seg_data.shape[1])
            z_start = max(z_center - half_size_z, 0)
            z_end = min(z_start + block_size[2], seg_data.shape[2])

            # Correct the start positions if the block exceeds the mask dimensions
            if x_end - x_start < block_size[0]: x_start = x_end - block_size[0]
            if y_end - y_start < block_size[1]: y_start = y_end - block_size[1]
            if z_end - z_start < block_size[2]: z_start = z_end - block_size[2]
              # Ensure adjustments did not result in negative start values
            x_start, y_start, z_start = max(x_start, 0), max(y_start, 0), max(z_start, 0)
            # Extract the 3x3x3 block
            
            # Check if the block contains only 0s
            block = seg_data[x_start:x_start+block_size[0], y_start:y_start+block_size[1], z_start:z_start+block_size[2]]
            if np.all(block == 0):
                block_found = True
                neg_output_coord_list.append((x_start, y_start, z_start))
                success_count += 1
    return neg_output_coord_list
    # else:
    #         assert filtered_separate_segmentation_masks is not None, "filtered_separate_segmentation_masks is None"
    #         for i in range(len(filtered_separate_segmentation_masks)):
    #             block 
    #         block = seg_data[start_x:start_x+block_size[0], start_y:start_y+block_size[1], start_z:start_z+block_size[2]]
    #         if np.all(block != 0):

    #             block_found = True
    #             if x_end - start_x < block_size[0]: start_x = x_end - block_size[0]
    #             if y_end - start_y < block_size[1]: start_y = y_end - block_size[1]
    #             if z_end - start_z < block_size[2]: start_z = z_end - block_size[2]

    #             pos_ooutput_coord_list.append((start_x, start_y, start_z))
    #             success_count += 1
    #         attempt += 1
    # if negative:
    #     return neg_output_coord_list
    # else:
    #     return pos_ooutput_coord_list

    # return None, None, attempt  # Return None if no block found after max attempts
    # return neg_output_coord_list, pos_ooutput_coord_list
def sample_pos_block(seg_data, local_max_points_coordinate, filtered_separate_segmentation_masks, block_size=(3, 3, 3)):
    # Find the indices where the value is 1
    pos_xyz_coordinates = np.zeros_like(seg_data)
    pos_output_coord_list = []
    for i in range(len(filtered_separate_segmentation_masks)):
        pos_xyz_coordinates[np.where(filtered_separate_segmentation_masks[i] == 1)] = 1
        # The returned coordinates are in the form of (x_indices, y_indices, z_indices)
        # To get a list of (x, y, z) tuples:
    local_max_points_coordinate_intersect_pos_coord = list(set([tuple(inner_list) for inner_list in local_max_points_coordinate]).intersection(set(pos_xyz_coordinates)))
    for index, (x_center, y_center, z_center) in enumerate(local_max_points_coordinate_intersect_pos_coord):
        # Calculate half sizes for each dimension, adjusting for even sizes
        half_size_x = block_size[0] // 2
        half_size_y = block_size[1] // 2
        half_size_z = block_size[2] // 2

        # Calculate start and end positions ensuring they are within the valid range
        x_start = max(x_center - half_size_x, 0)
        x_end = min(x_start + block_size[0], seg_data.shape[2])  # Notice shape index for width is 2
        y_start = max(y_center - half_size_y, 0)
        y_end = min(y_start + block_size[1], seg_data.shape[1])  # Notice shape index for height is 1
        z_start = max(z_center - half_size_z, 0)
        z_end = min(z_start + block_size[2], seg_data.shape[0])  # Notice shape index for depth is 0

        # Correct the start positions if the block exceeds the segmentation data dimensions
        if x_end - x_start < block_size[0]: x_start = x_end - block_size[0]
        if y_end - y_start < block_size[1]: y_start = y_end - block_size[1]
        if z_end - z_start < block_size[2]: z_start = z_end - block_size[2]

        # Now, x_start, y_start, z_start, x_end, y_end, and z_end define a valid block within the segmentation data dimensions.
        # You can use these to extract or manipulate blocks within seg_data.
        # For example, extracting a block:
        # block = seg_data[z_start:z_end, y_start:y_end, x_start:x_end]

        # Ensure adjustments did not result in negative start values
        x_start, y_start, z_start = max(x_start, 0), max(y_start, 0), max(z_start, 0)
        pos_output_coord_list.append((x_start, y_start, z_start))
    return pos_output_coord_list

def get_suv_block(suv_data, position, block_size=(3, 3, 3)):
    start_x, start_y, start_z = position
    end_x = start_x + block_size[0]
    end_y = start_y + block_size[1]
    end_z = start_z + block_size[2]
    block = suv_data[start_x:end_x, start_y:end_y, start_z:end_z]
    assert block.shape == block_size, f"Block shape is wrong and currently has shape of : {block.shape}"
    return block

# block, start_position, attempts = sample_3x3x3_block_full_of_0s(seg_data)

# block, start_position, attempts
