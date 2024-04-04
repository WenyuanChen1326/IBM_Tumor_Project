import numpy as np
# np.random.seed(0)  # Ensure reproducibility


def sample_neg_block(seg_data, local_max_points_coordinate_without_pos_coordinate, block_size=(3, 3, 3), sample_size=10, negative=True):
    if len(local_max_points_coordinate_without_pos_coordinate) == 0:
        return []
    max_attempts = 100
    # attempt = 0
    # block_found = False
    adjusted_smaple_size = min(sample_size, len(local_max_points_coordinate_without_pos_coordinate))
    neg_output_coord_list = []
    success_count = 0
    while success_count < adjusted_smaple_size:
        # print(f"success_count: {success_count}")
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
            if negative:
                if np.all(block == 0):
                    block_found = True
                    neg_output_coord_list.append((x_start, y_start, z_start))
                    success_count += 1
                    attempt += 1
            else:
                block_found = True
                neg_output_coord_list.append((x_start, y_start, z_start))
                success_count += 1
                attempt += 1

            
    return neg_output_coord_list

def sample_neg_block_with_suv_threshold(seg_data, suv_data, local_max_points_coordinate_without_pos_coordinate, block_size=(3, 3, 3), sample_size=10, negative=True, max_suv_min_threshold = 5):
    if len(local_max_points_coordinate_without_pos_coordinate) == 0:
        return []
    max_attempts = 100
    # attempt = 0
    # block_found = False
    adjusted_smaple_size = min(sample_size, len(local_max_points_coordinate_without_pos_coordinate))
    neg_output_coord_list = []
    neg_output_suv_bock_list = []
    success_count = 0
    while success_count < adjusted_smaple_size:
        # print(f"success_count: {success_count}")
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
            if negative:
                if np.all(block == 0):
                    if max_suv_min_threshold is None:
                        block_found = True
                        neg_output_coord_list.append((x_start, y_start, z_start))
                        success_count += 1
                        attempt += 1
                    else:
                        suv_block = suv_data[x_start:x_start+block_size[0], y_start:y_start+block_size[1], z_start:z_start+block_size[2]]
                        if np.max(suv_block) >= max_suv_min_threshold:
                            block_found = True
                            neg_output_coord_list.append((x_start, y_start, z_start))
                            neg_output_suv_bock_list.append(suv_block)
                            success_count += 1
                            attempt += 1
            else:
                block_found = True
                neg_output_coord_list.append((x_start, y_start, z_start))
                success_count += 1
                attempt += 1
    return neg_output_suv_bock_list

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
