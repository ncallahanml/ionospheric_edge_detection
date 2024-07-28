import os

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from tqdm import tqdm

def pad_axis(arr, expected_size, dtype=np.uint8, axis=0):
    shape_mismatch = expected_size - arr.shape[axis]
    left_pad = shape_mismatch // 2
    
    if shape_mismatch > 0:
        right_pad = shape_mismatch - left_pad
        axis_pad = (left_pad, right_pad)
        full_pad = [(0, 0) if i != axis else axis_pad for i in range(arr.ndim)]
        arr = np.pad(arr, tuple(full_pad), mode='constant', constant_values=0)
    elif shape_mismatch < 0:
        left_pad = -shape_mismatch // 2
        right_pad = -shape_mismatch - left_pad
        arr = arr[:,left_pad:-right_pad]
        
    if arr.dtype != dtype:
        raise TypeError(
            f'output array unexpected dtype, expected {arr.dtype} recieved {dtype}'
        )
    
    if arr.shape[axis] != expected_size:
        raise ValueError(
            f'{arr.shape[axis]} Mismatches Expected {axis} Dimension of {expected_size}'
        )
    return arr
    
def pad_img(img, expected_shape=(1440, 300), dtype=np.uint8):
    """
    Raw input data has inconsistent size, very close but not precisely
    the intended (1440, 300) size. This pads the image to make
    it exactly (1440, 300) but does so evenly on both sides, if required.
    """
    if len(expected_shape) != img.ndim:
        raise ValueError(
            f'Mismatched dimensions, expected {len(expected_shape)} received {img.ndim} for shape {img.shape}'
        )
    for i in range(img.ndim):
        img = pad_axis(img, expected_shape[i], axis=i, dtype=dtype)
    return img

def cut_half(img, expected_size=1440):
    """ Simple preprocessing for image, could add additional adjustments here """
    if expected_size:
        if img.shape[0] != expected_size:
            raise ValueError(
                f'Mismatch with width, dim 0 of {img.shape} != {expected_size}'
            )
        if expected_size % 2:
            raise ValueError(
                f'Width must be even, received {expected_size}'
            )
    img = img[expected_size // 2:,:]
    return img

def cat_date_imgs(
    parent_dir='raw_data/', 
    filter_fn=None, 
    max_iter=None, 
    read_lib='pandas',
    dtype=np.uint16, 
    height_start=0, 
    time_start='12:00',
    apply_fn=None,
    split_idx=-1,
):
    """
    Function to load and preprocess a directory of raw data stored as spatial 2D NumPy arrays.
    Arrays are loaded from files and concatenated into a single 3D NumPy array.
    
    ---
    Args:
    
    **parent_dir** : `str` | default 'raw_data/'
    - Directory full of raw image files
    
    **filter_fn** : `Callable` or `None` | default None
    - Function applied to filter down input images from `parent_dir`
    - Defaults to filtering directory files out if they are not CSVs
    
    **max_iter** : `int` or `None` | default None
    - If integer, the maximum number of images to process from the directory
    - If None, the entire directory will be processed
    - Files are processed in sorted order
    - Files that are filtered out by `filter_fn` are not considered an iteration
    
    **read_lib** : `str` | default 'pandas'
    - Library to use for reading array data from CSV
    
    **dtype** : `np.type` or `Tuple[np.type]` | default np.uint16
    - Type for storing images
    - If np.type, used for both original processing and final 3d array output
    - If tuple, must be length 2
        - Where first argument is initial dtype
        - The second argument is the output dtype
    - Should stay np.uint16 for raw data, can be np.uint8 prior to ~2022
    
    **apply_fn** : `Callable` or `None` | default None
    - Function to apply to each image prior to stacking
    - Takes as input numpy image array with dtype defined by `dtype`
    - Returns the same shape numpy image array
    - Output dtype must be convertible to second dtype defined by `dtype`
    
    **split_idx** : `int` | default -1
    - Index of the date in the name of loaded file once split by underscores
    - -1 signifies the date is bracketed between
        - Underscore on the left
        - The file extension on the right
    
    ---
    Returns :
    
    **date_img_xarr** : `xr.DataArray`
    - 3 dimensional array with coords (date, time, height)
    - Shape is (n, expected_shape[0], expected_shape[1])
        - n is the length of images in the `parent_dir`
        - Is reduced from total files based on `filter_fn` and `max_iter`
    - Data type is determined by output dtype from `dtype`
    - Will be raw data if `apply_fn` is None
    
    """
    # fixed args, shouldn't be modified without other code changes
    expected_shape = (720, 300)
    start_time = '12:00:00'
    start_height = 0
    
    if isinstance(dtype, tuple):
        if not (dtype_len := len(dtype)) == 2:
            raise ValueError(
                f'If passing a tuple for `dtype`, must be length 2, not {dtype_len}'
            )
        in_dtype, out_dtype = dtype
    elif isinstance(dtype, np.type):
        in_dtype, out_dtype = (dtype, dtype)
    else:
        raise TypeError(
            f'Expected `tuple` or `np.type` for `dtype`, not {type(dtype)}'
        )
    
    img_list, stat_list = list(), list()

    if filter_fn is None:
        # default filter down to CSVs only
        filter_fn = lambda x : x.endswith('.csv')
    file_paths = sorted(filter(filter_fn, os.listdir(parent_dir)))
    max_paths = len(file_paths) if max_iter is None else min(max_iter, len(file_paths))
    
    for file_path in tqdm(file_paths[:max_paths]):
        full_path = os.path.join(parent_dir, file_path)
        
        file_path_date = file_path.split('_')[split_idx].replace('.csv','')
        try:
            date = pd.to_datetime(file_path_date)
        except pd.errors.ParserError:
            raise ValueError(f'Split returned invalid date')

        if read_lib == 'pandas':
            img = pd.read_csv(full_path)
            if not np.all(img >= 0):
                raise ValueError(
                    f'Input data of shape {img.shape} contains only zeros'
                )
            if not np.all(img <= np.iinfo(in_dtype).max):
                raise ValueError(
                    f'Unacceptable input dtype {in_dtype} passed, value {np.amax(img.ravel())} exceeds size limits'
                )
            img = img.to_numpy(dtype=in_dtype)
        elif read_lib == 'numpy':
            img = np.genfromtxt(full_path, delimiter=',').astype(in_dtype)
        elif read_lib == 'modin':
            raise NotImplementedError('Modin currently untested, not in requirements')
            img = md.read_csv(full_path)
        else:
            raise ValueError(
                f'Unrecognized literal {read_lib} for variable `read_lib`, accepts "pandas" or "numpy"'
            )
        
        # standardizes width to expected size
        img = pad_img(
            img, 
            expected_shape=(expected_shape[0] * 2, expected_shape[1]), 
            dtype=in_dtype,
        )
        # trims to only 12 daytime hours instead of full 24
        img = cut_half(
            img, 
            expected_size=expected_shape[0] * 2,
        )
        # verifies dimensions all match before stacking
        if img.shape != expected_shape:
            raise ValueError(
                f'Expected image shape {expected_shape}, received {img.shape}'
            )
        # applies function in 2d
        if apply_fn is not None:
            img = apply_fn(img)
            
        img_list.append((date.to_pydatetime(), img))
        
    dates, imgs = zip(*img_list)

    # image horizontal labels
    times = pd.timedelta_range(
        start=start_time, 
        end='23:59:00', 
        freq='1min',
    )
    # image vertical labels
    heights = np.arange(
        start_height, 
        10 * expected_shape[1], 
        10,
    )

    date_img_xarr = xr.DataArray(
        np.stack(imgs, axis=0, dtype=out_dtype),
        coords={
            'date' : list(dates),
            'time' : times,
            'height' : heights,
        },
        dims=['date','time','height'],
    )
    return date_img_xarr

def mad(t, min_dev=.05):
    # median absolute deviation
    median = np.median(t, axis=(0, 1), keepdims=True)
    abs_devs = np.abs(t - median)
    max_median = max(
        np.median(abs_devs, axis=(0, 1), keepdims=True), 
        min_dev,
    )
    mad = abs_devs / max_median
    if t.shape != mad.shape:
        raise ValueError(
            f'Function internal error, input shape {t.shape} mismatches output shape {mad.shape}'
        )
    return mad
