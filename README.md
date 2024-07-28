# Image Loading

Images are loaded from CSVs within a parent directory. By default only '.csv' files are loaded, although
other file extensions with CSV formatting can be allowed. Files are loaded in sorted order.

Images are expected to have a name separated by underscores within the file path. The location of
the date in the array of split file name literals is expected to be at index 1 (2nd item) based
on sample data filenames. This can be adjusted with the `split_idx` argument. Dates are expected
to be in the format `%Y-%m-%d`, consistent with sorting order and the `pandas` library defaults.

# Image Trimming

Currently full 24 hour images in grayscale are expected, with a height of 300 px, leading to the shape
(1440, 300). Some of the input images have had slight deviations from this, such as being +-2 pixels off,
so a very slight padding is applied to ensure all sizes are (1440, 300).

Due to the latter 12 hours of each file containing the only consistent data, before storing the raw data
the first half of the image is dropped, resulting in a shape of (720, 300). After cutting off the first half
of the image, an image-by-image preprocessing function can be applied.

Once the full list of (720, 300) images is completely loaded into memory, the images are stacked along the
0th dimension into shaped (`n`, 720, 300) where `n` is the number of loaded images. After stacking, an 
Xarray DataSet is used to wrap the dimensions with appropriate labels, with coordinates (date, time, height)
respectively.

# Threshold Processing

When processing each image, the current default is to also trim off the first and last hour of data,
as well as trimming the vertical axis to remove regions where LSTIDs are not regularly observered. 
The function for median absolute deviation rescaling is applied during the image loading process. 
Additionally, gaussian blurring is essential for robust results.

After all of these image wide processing steps are taken, additional outlier trimming is used for rescaling
thresholds. Given a number of occurrences, the largest pixel values are trimmed back to the maximum value
corresponding to the location of the number of occurences in the cumulative histogram. This saves
the threshold rescaling from being heavily affected by a few outlying pixels, which are effectively
discarded.

The images are then discretized into a fixed number of populated values. The minimum is first subtracted from
the image array to establish the new minimum at zero, and the remaining values are rounded to the nearest discrete bin. 
The number of discrete bins is a parameter set by the user, and defaults to 30.

# Threshold Measurement

For each discrete bin represented in the image, values lower than the bin are removed. This results in a minimum
line tracking the edge for that threshold, the vertical values of minimum height measured data are stored
in the corresponding bin. When the minimum for the threshold is at or below 0, or at or above the maximum
height represented, this is stored as `np.nan` in the threshold array.

Once all thresholds have been compiled, they are stacked together to form a distribution for every column in
the image with a variable number of NaN values in the column. Ignoring NaNs, the percentile for the array is
taken as the definitive detection for that column. Multiple percentiles can be returned and plotted at the 
same time, but the current functionality is to only return one selected array per day.

All the individual edges are returned as a DataFrame with a common index. These are not directly smoothed before
being returned, although the image transformations and other hyperparameters for the detection have a
significant influence on the noisiness of the returned lines.