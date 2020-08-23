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

def vec2mat(tri_vec):
    '''
    Transfer the upper-triangle vector without diagonal element
    to functional connectivity matrix

    parameters
    ----------
    tri_vec: the upper-triangle vector of the functional connectivity, np.array.
                 the shape is (L, N*(N-1) / 2)

    return
    ------
    connectivity: the functional connectivity, np.array, (L, N, N).
                  where L is the number of subjects, N is the number of ROIs.

    '''
    n_roi = int(np.sqrt(tri_vec.shape[1] * 2)) + 1
    n_subject = tri_vec.shape[0]
    connectivity = np.zeros((n_subject, n_roi, n_roi))
    triu_idx = np.triu_indices_from(connectivity[0], 1)
    tril_idx = np.tril_indices_from(connectivity[0], -1)
    for k in range(n_subject):
        connectivity[k][triu_idx], connectivity[k][tril_idx] = tri_vec[k], tri_vec[k]
    return connectivity


def get_functional_connectivity_index(feature_index, n_roi, T=0):
    '''
    Transfer the upper-triangle vector index (without diagonal element)
    to functional connectivity (FC) index
    The vector index and FC index both start with 0.

    parameters
    ----------
    feature_index: the index of the upper-triangle vector, list. The element is int.
    n_roi: the number of the ROIs, int.
    T: the flag whether append the transpose position connectivity. int
       if T=0, the result will include both (ROI_i, ROI_j) and (ROI_j, ROI_i)
       else if T=1, the result will only include (ROI_i, ROI_j) with j > i

    return
    ------
    FC_index: the functional connectivity index corresponded to the feature_index, list.
              the element is tuple (ROI_i, ROI_j) with n_roi > j > i >= 0
              if T=0, the result will also include ((ROI_j, ROI_i))
    '''

    assert np.max(feature_index) <= n_roi * (n_roi - 1) / 2 - 1
    FC_index = []
    for feat_index in feature_index:
        # assume k is row number of the feat_index and k >= 0
        # it satisfied k*(2*n_roi - k - 1) / 2 - 1 < feat_index <= (2*n_roi-k-2)(k+1)/2 - 1

        minus_b1, delta1 = 2 * n_roi - 1, (2 * n_roi - 1) ** 2 - 4 * 2 * (feat_index + 1)
        minus_b2, delta2 = 2 * n_roi - 3, (2 * n_roi - 3) ** 2 - 4 * 2 * (feat_index - n_roi + 2)

        k1, k2 = (minus_b1 - np.sqrt(delta1)) / 2, (minus_b1 + np.sqrt(delta1)) / 2
        k3, k4 = (minus_b2 - np.sqrt(delta2)) / 2, (minus_b2 + np.sqrt(delta2)) / 2

        range_1, range_2 = set(range(0, np.ceil(k1).astype(int))) | set(range(np.ceil(k2).astype(int), n_roi)), \
                           set(range(np.ceil(k3).astype(int), np.ceil(k4 + 1e-12).astype(int)))
        assert len(range_1 & range_2) == 1
        k = (range_1 & range_2).pop()

        m = int(feat_index - k * (2 * n_roi - k - 1) / 2 + k + 1)
        FC_index.append((k, m))
        if T == 0:
            FC_index.append((m, k))
    return FC_index


def get_feature_vector_index(FC_index, n_roi):
    '''
    Transfer the functional connectivity (FC) index
    to upper-triangle vector index (without diagonal element)
    The vector index and FC index both start with 0.

    parameters
    ----------
    FC_index: the index of the upper-triangle functional connectivity, list.
              The element is tuple (ROI_i, ROI_j) with j > i >= 0
    n_roi: the number of the ROIs, int. it starts with 0

    return
    ------
    feature_index: the upper-triangle vector index index corresponded to the FC_index, list.
                   the element is int.
    '''

    feature_index = []
    for (i, j) in FC_index:
        feat_index = i * (2 * n_roi - i - 1) // 2 + j - i - 1
        feature_index.append(feat_index)
    return feature_index


