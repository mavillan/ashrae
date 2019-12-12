import numpy as np
import pandas as pd
from itertools import count
import collections

def moving_average(data, window_size):
    """ 
    Computes moving average using discrete linear convolution 
    of two one dimensional sequences.
    Parameters
    ----------
        data (pandas.Series): independent variable
        window_size (int): rolling window size
    Returns
    ---------
            np.ndarray of linear convolution
    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, "same")


def anomaly_detector(y, window_size=10, sigma=2.0):
    """ 
    Identifies anomalies in TS data through rolling std
    Parameters
    ----------
    y: pandas.Series 
        independent variable
    window_size: int 
        rolling window size
    sigma : int 
        value for standard deviation
    Returns
    --------
        Dictionary {'standard_deviation': int, 'anomalies_dict': (index: value)}
        containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # calculate the variation in the distribution of the residual
    testing_std = residual.rolling(window_size).std()
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan,
                                  testing_std_as_df.loc[window_size - 1]).round(3).iloc[:,0].tolist()
    ret = [(index, y_i) for 
           index, y_i, avg_i, rs_i in zip(count(),y,avg_list,rolling_std) 
           if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))]
    return ret


def find_outliers(ts, window, threshold, periodic_borders=False):
    assert window%2 == 1, "Parameter 'window' must be odd."
    ts_lenght = len(ts)
    if periodic_borders:
        ts_ext = pd.concat([ts.tail(window//2), ts, ts.head(window//2)], ignore_index=True)
        rw = ts_ext.y.rolling(window, center=True)
        rw_median = (rw.median()
                     .loc[window//2:ts_lenght-1+window//2]
                     .reset_index(drop=True))
        rw_iqr = ((rw.quantile(0.75)-rw.quantile(0.25))
                  .loc[window//2:ts_lenght-1+window//2]
                  .reset_index(drop=True))
    else:
        rw = ts.y.rolling(window, center=True)
        rw_median = (rw.median()
                     .interpolate(method="linear", limit_direction="both"))
        rw_iqr = ((rw.quantile(0.75)-rw.quantile(0.25))
                  .interpolate(method="linear", limit_direction="both"))
    outlier_idx1 = ts.query("y > @rw_median + @threshold*@rw_iqr").index
    outlier_idx2 = ts.query("y < @rw_median - @threshold*@rw_iqr").index
    return outlier_idx1.union(outlier_idx2)