import pandas as pd
import xarray as xr
import numpy as np
import joblib

######################

import matplotlib.pyplot as plt
import seaborn as sns
# import pandas as pd
# import numpy as np

import warnings
# import joblib
import math
import os

from scipy.ndimage import gaussian_filter
from IPython.display import clear_output

######################

from data_loading import (
    cat_date_imgs,
    mad,
)
# from utils import DateIter
from threshold_edge_detection import (
    # lowess_smooth,
    measure_thresholds,
)

class DateIter():
    def __init__(self, xarr):
        if isinstance(xarr, xr.DataArray):
            self.data = xarr
        elif isinstance(xarr, str):
            self.data = joblib.load(xarr)
        else:
            raise TypeError(f'Unexpected type {type(xarr)} for input array')
        
        return
    
    def get_date(self, date, raise_missing=True):
        date = pd.to_datetime(date)

        try:
            xarr = self.data.sel(date=date)
        except KeyError as ke:
            if raise_missing:
                raise ke
            else:
                return

        return xarr
    
    def iter_dates(self, dates, skip_missing=False, **get_kwargs):
        for date in dates:
            if isinstance(date, tuple):
                start_date, end_date = date
                dates = pd.date_range(start=start_date, end=end_date)
                for date, arr in self.iter_dates(dates, skip_missing=skip_missing, **get_kwargs):
                    yield date, arr
            else:
                if skip_missing:
                    try:
                        yield date, self.get_date(date, **get_kwargs)
                    except KeyError:
                        continue
                else:
                    yield date, self.get_date(date, **get_kwargs)
                
    def iter_all(self):
        return self.iter_dates(self.data.indexes['date'])
    
def save_wrap(save_dir, fmt='%Y-%m-%d', ext='.png', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    def wrapped(date):
        date_str = pd.to_datetime(date).strftime(fmt)
        file_path = os.path.join(save_dir, date_str + ext)
        plt.savefig(file_path, **kwargs)
        return
    return wrapped

def plot_day(arr_df, data, edge_line, date, plt_save_path, plot):
    if plt_save_path is not None:
        save_plt = save_wrap(plt_save_path)
    else:
        save_plt = None

    fig, ax = plt.subplots(1, 1, figsize=(15,8))
    plt.title(f'| {date} |')

    sns.heatmap(
        arr_df, 
        robust=True, 
        cbar=False, 
        ax=ax,
    )
    ax.invert_yaxis()

    sns.lineplot(
        data=data, 
        alpha=.75, 
        dashes=False, 
        ax=ax, 
        palette='light:b',
    )
    sns.lineplot(
        data=edge_line, 
        x='Time', 
        y='Height', 
        color='white', 
        alpha=1, 
        dashes=False, 
        ax=ax,
    )

    if save_plt is not None:
        save_plt(date)

    plt.show() if plot else plt.clear()
    return

def trim_edges(arr, x_trim, y_trim):
    xl_trim, xr_trim = x_trim if isinstance(x_trim, (tuple, list)) else (x_trim, x_trim)
    yl_trim, yr_trim = x_trim if isinstance(y_trim, (tuple, list)) else (y_trim, y_trim)
    xr = math.floor(xl_trim * arr.shape[0])
    xl = math.floor(xr_trim * arr.shape[0])
    yr = math.floor(yl_trim * arr.shape[1])
    yl = math.floor(yr_trim * arr.shape[1])

    arr = arr[xr:-xl, yr:-yl]
    return arr

def run_edge_detect(
    dates,
    date_iter,
    x_trim=.08333,
    y_trim=.08,
    sigma=4.2, # 3.8 was good
    qs=[.4],
    occurence_n = 60,
    i_max=30,
    plot=True,
    clear_every=100,
    plt_save_path=None,
    csv_save_path=None,
    thresh=None,
):       
    final_edge_list = list()
    if dates == 'all':
        date_gen = date_iter.iter_all()
    else:
        date_gen = date_iter.iter_dates(dates, raise_missing=False)

    for i, (date, arr) in enumerate(date_gen):
        if arr is None:
            warnings.warn(f'Date {date} has no input')
            continue
            
        if not i % clear_every:
            clear_output()
        
        arr = trim_edges(arr, x_trim, y_trim)
        heights, times = arr.coords['height'], arr.coords['time']

        # next lines convert from xarray arr to numpy arr
        arr = np.nan_to_num(arr, nan=0)
        arr = gaussian_filter(arr.T[::1,:], sigma=(sigma, sigma))  # [::-1,:]
        
        med_lines, min_line, minz_line = measure_thresholds(
            arr[::1],
            qs=qs, 
            occurrence_n=occurence_n, 
            i_max=i_max,
        )

        data = pd.DataFrame(
            np.array(med_lines).T,
            index=(date + times).dt.strftime('%H:%M'),
            columns=qs,
        ).reset_index(
            names='Time',
        )
        if thresh is None:
            edge_line = pd.DataFrame(
                min_line, 
                index=(date + times).dt.strftime('%H:%M'), 
                columns=['Height'],
            ).reset_index(
                names='Time'
            )
        elif isinstance(thresh, dict):
            edge_line = (
                data[['Time', thresh[date]]]
                .rename(columns={thresh[date] : 'Height'})
            )
        elif isinstance(thresh, float):
            edge_line = (
                data[['Time', thresh]]
                .rename(columns={thresh : 'Height'})
            )
        else:
            raise ValueError(
                f'Threshold {thresh} of type {type(thresh)} is invalid'
            )

        final_edge_list.append(
            pd.Series(min_line.squeeze(), index=times, name=date)
        )

        if plot or plt_save_path is not None:
            arr_df = pd.DataFrame(
                arr,
                index=heights,
                columns=times,
            )
            plot_day(arr_df, data, edge_line, date, plt_save_path, plot)
            
    final_edge_df = pd.concat(final_edge_list, axis=1)
    if csv_save_path:
        final_edge_df.to_csv(csv_save_path)
    
    return final_edge_df

# directory full of raw count CSVs
# path for serialized version of preprocessed CSV data, volumetric

def complete_edge_detections(
    parent_dir,
    csv_path=None,
    date_list='all',
    q=.4,
    clear_every=10,
    split_idx=1,
):
    if not os.path.exists(parent_dir) or not os.path.isdir(parent_dir):
        raise ValueError(
            f'Must pass a valid path to a directory with CSVs'
        )
    
    if not isinstance(q, float):
        raise TypeError(
            f'Expected `q` to be a float, received type {type(q)}'
        )
    
    date_img_xarr = cat_date_imgs(
        parent_dir=parent_dir,
        dtype=(np.uint16, np.float32),
        apply_fn=mad,
        split_idx=split_idx,
    )

    date_iter = DateIter(date_img_xarr)

    detected_edge_df = run_edge_detect(
        date_list,
        date_iter,
        qs=[q],
        clear_every=clear_every,
        csv_save_path=None,
        plot=False,
        thresh=q,
    )

    reindexed_edge_df = (
        detected_edge_df
        .reindex(
            pd.timedelta_range(
                start='12:00:00', 
                end='23:59:00', 
                freq='1min',
            ),
            axis=0,
        )
        .fillna(0)
    )

    if csv_path is not None:
        reindexed_edge_df.to_csv(csv_path)

    return reindexed_edge_df

def plot_date(date_iter, edge_df, date, side_trim=.08):
    if isinstance(edge_df, str):
        edge_df = pd.read_csv(edge_df, index_col=0)
        edge_df.columns = pd.to_datetime(edge_df.columns)
    
    date = pd.to_datetime(date)
    arr = date_iter.get_date(date)
    x_trim = math.floor(side_trim * arr.shape[0])
    y_trim = math.floor(side_trim * arr.shape[1])
    arr = arr[x_trim:-x_trim, y_trim:-y_trim]
    
    edge = edge_df[date]
    
    fig, ax = plt.subplots(1, 1, figsize=(15,8))
    plt.title(date)
    ax.axis('off')
    sns.heatmap(arr.T, robust=True, cbar=False, ax=ax)
    ax.invert_yaxis()

    sns.lineplot(
        data=edge, 
        color='white', 
        alpha=1, 
        dashes=False, 
        ax=ax,
    )    
    plt.show()
    return