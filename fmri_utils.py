from nilearn import connectome
import numpy as np


def get_functional_connectivity(total_time_series):
    '''

    Get the functional connectivity matrix (diagonal elements are 0) from the roi_signals

    parameters
    ----------
    total_time_series: the roi_signals, list (length L).
                       the elements are np.array with shape (T, N)
                       where T is the time series step, N is the number of ROIs

    return
    ------
    connectivity: the functional connectivity matrix from the roi_signals, np.array.
                  the shape is (L, N, N)

    '''
    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    connectivity = conn_measure.fit_transform(total_time_series)
    diagonal = np.diag_indices(connectivity.shape[1])
    for i in range(connectivity.shape[0]):
        connectivity[i][diagonal] = 0
    connectivity = np.arctanh(connectivity)
    return connectivity


def split_time_series(time_series, k=3):
    '''

    Split the roi_signals into k segmented part

    parameters
    ----------
    time_series: the roi_signal, np.array. Shape is (T, N)
                 where T is the time series step, N is the number of ROIs
    k: the times to split the roi_signal, int. Default is 3.

    return
    ------
    split_roi_signals: the split roi_signals in temporary order, list (length k).
                       the element is np.array with shape (T//k, N) or (T//k + T%k, N)
    '''
    return np.array_split(time_series, k, axis=0)


def slide_window_of_time_series(time_series, window_size=0.2, step=1):
    '''

    Get the temporary sliding window-size roi_signals

    parameters
    ----------
    time_series: the roi_signal, np.array. Shape is (T, N)
                 where T is the time series step, N is the number of ROIs
    window_size: the window size ratio to time length, float (0, 1).
                 The real window size is int(T * window_size)
    step: the slide step, int.

    return
    ------
    slide_window_series: the slide roi-signals, list.
                         the elements is np.array with shape (window_size*T, N)
    '''
    assert 0 < window_size <= 1

    time_length = time_series.shape[0]
    window = int(window_size * time_length)
    slide_window_series = []

    for start in range(0, time_length, step):
        end = start + window
        if end > time_length:
            end = time_length
        if end == start + 1:
            break
        window_series = time_series[start: end]
        slide_window_series.append(window_series)

    return slide_window_series


def get_window_time_series(time_series, window_size=0.6, start=None):
    '''

    Get the window-size roi_signals

    parameters
    ----------
    time_series: the roi_signal, np.array. Shape is (T, N)
                 where T is the time series step, N is the number of ROIs
    window_size: the window size ratio to time length, float (0, 1).
                 The real window size is int(T * window_size)
    start: the window start position, int. Default is None.
           if None, the start position will be random.

    return
    ------
    window_series: the window-size roi-signals, np.array.
                   the shape is (window_size*T, N)
    '''
    time_length = time_series.shape[0]
    if start is not None:
        start_pos = start
    else:
        last_start_pos = int((1 - window_size) * time_length)
        start_pos = int(np.round(np.random.rand(1) * last_start_pos)[0])
    end_pos = int(time_length * window_size) + start_pos
    window_series = time_series[start_pos:end_pos]
    return window_series


def mat2vec(connectivity):
    '''
    Transfer the functional connectivity matrix to upper-triangle vector
    without diagonal element

    parameters
    ----------
    connectivity: the functional connectivity, np.array, (L, N, N).
                  where L is the number of subjects, N is the number of ROIs.


    return
    ------
    feature_vec: the upper-triangle vector of the functional connectivity, np.array.
                 the shape is (L, N*(N-1) / 2)

    '''
    idx = np.triu_indices_from(connectivity[0], 1)
    vec_list = [mat[idx] for mat in connectivity]
    feature_vec = np.vstack(vec_list)
    return feature_vec



