import h5py # type: ignore[import-untyped]
import hdf5plugin

import numpy as np
import gc

def load_hdf5(dir):

    data_train = None
    with h5py.File(dir, "r", swmr=True, locking=False) as f:
        first_item_name = list(f.keys())[0]
        data = f[first_item_name]
        data_train,_ = extract_random_block_3d(data,(384,384,384))
        f.close()

    gc.collect()
    return data_train


def extract_random_block_3d(array_3d, block_size):
    """
    Randomly extracts a block from a 3D NumPy array.

    If any dimension of the requested `block_size` is larger than the array's
    corresponding dimension, that block dimension will be capped to the
    array's dimension size. This means the extracted block will never be
    larger than the array itself in any dimension.

    Args:
        array_3d (np.ndarray): The input 3D NumPy array.
        block_size (tuple): A tuple (depth, height, width) specifying the
                            desired dimensions of the block to extract.

    Returns:
        np.ndarray: The randomly extracted block.
        tuple: The (z_start, y_start, x_start) coordinates of the top-left-front
               corner of the extracted block.
        tuple: The effective block_size (depth, height, width) that was actually extracted
               after applying any capping.
    """
    array_shape = array_3d.shape
    requested_block_depth, requested_block_height, requested_block_width = block_size

    # --- Implement Capping Logic ---
    # The effective block dimension is the minimum of the requested size and the array's size
    effective_block_depth = min(requested_block_depth, array_shape[0])
    effective_block_height = min(requested_block_height, array_shape[1])
    effective_block_width = min(requested_block_width, array_shape[2])

    effective_block_size = (effective_block_depth, effective_block_height, effective_block_width)

    # Calculate the maximum possible starting coordinates for each dimension
    # If effective_block_dim == array_dim, max_start will be 0.
    max_z_start = array_shape[0] - effective_block_depth
    max_y_start = array_shape[1] - effective_block_height
    max_x_start = array_shape[2] - effective_block_width

    # Ensure max_start is not negative in case of very small arrays/large block_size requests
    # (Though min() should handle this, it adds robustness for edge cases)
    if max_z_start < 0: max_z_start = 0
    if max_y_start < 0: max_y_start = 0
    if max_x_start < 0: max_x_start = 0


    # Randomly choose starting coordinates
    # np.random.randint(low, high_exclusive)
    z_start = np.random.randint(0, max_z_start + 1)
    y_start = np.random.randint(0, max_y_start + 1)
    x_start = np.random.randint(0, max_x_start + 1)

    # Calculate end coordinates for slicing
    z_end = z_start + effective_block_depth
    y_end = y_start + effective_block_height
    x_end = x_start + effective_block_width

    # Extract the block using slicing
    #extracted_block = array_3d[z_start:z_end, y_start:y_end, x_start:x_end]

    extracted_block = np.zeros(block_size, dtype='int32')
    array_3d.read_direct(extracted_block, np.s_[z_start:z_end, y_start:y_end, x_start:x_end], np.s_[0:effective_block_depth,0:effective_block_height,0:effective_block_width])

    return extracted_block, (z_start, y_start, x_start)
