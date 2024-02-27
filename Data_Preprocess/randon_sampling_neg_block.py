import numpy as np
import numpy as np

# Assume `array` is your binary 3D array. For demonstration, create a dummy 3D array.
# This example array is mostly 0s for demonstration; in real scenarios, adjust the array accordingly.
np.random.seed(0)  # Ensure reproducibility
# array = np.random.choice([0, 1], size=(400, 400, 326), p=[0.98, 0.02])  # Example with mostly 0s

def sample_3x3x3_block_full_of_0s(seg_data, block_size=(3, 3, 3)):
    max_attempts = 10000
    attempt = 0
    block_found = False

    while attempt < max_attempts and not block_found:
        # Generate random starting indices
        start_x = np.random.randint(0, seg_data.shape[0] - 2)  # -2 to include the end index for a 3x3x3 block
        start_y = np.random.randint(0, seg_data.shape[1] - 2)
        start_z = np.random.randint(0, seg_data.shape[2] - 2)

        # Extract the 3x3x3 block
        block = seg_data[start_x:start_x+block_size[0], start_y:start_y+block_size[1], start_z:start_z+block_size[2]]

        # Check if the block contains only 1s
        if np.all(block == 0):
            block_found = True
            return block, (start_x, start_y, start_z), attempt + 1  # Return block, starting position, and attempts

        attempt += 1

    return None, None, attempt  # Return None if no block found after max attempts

# Attempt to sample a 3x3x3 block full of 1s from the array
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
