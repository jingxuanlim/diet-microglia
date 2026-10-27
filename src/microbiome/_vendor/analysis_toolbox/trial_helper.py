# Vendored from analysis_toolbox (trial_helper.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


import numpy as np
from tqdm import tqdm_notebook as tqdm


def construct_trig_window(before_trig, after_trig, fs):

    from microbiome._vendor.analysis_toolbox.utils import gain_arange

    window = gain_arange(before_trig, after_trig, fs)
    t_axis = window / fs

    return t_axis, window


def ensure_valid_index(indices, ind_min=-np.inf, ind_max=np.inf):

    valid = np.logical_and(
        np.all(indices <= ind_max, axis=1), np.all(indices >= ind_min, axis=1)
    )

    return indices[valid], valid


def event_triggered(window, event_ind, max_ind=None):

    # swim triggered windows at ephys rate
    trig_ind = window + event_ind[np.newaxis, :].transpose()

    if max_ind is not None:

        # remove indices above total length
        trig_ind, trig_ind_valid = ensure_valid_index(
            trig_ind, ind_min=0, ind_max=max_ind
        )

        return trig_ind.astype("int"), trig_ind_valid.astype("bool")

    else:
        return trig_ind


def find_trial_indices(trial_series):

    trial_inds = []

    for trial in tqdm(np.unique(trial_series)[np.unique(trial_series) >= 0]):
        trial_inds.append(np.where(trial_series == trial)[0])

    return trial_inds


def divide_into_trials(trial_num, vars_dict, trial_inds_name="trial_inds"):

    trial_vars = divide_into_trials_dict(trial_num, vars_dict)
    trial_vars[trial_inds_name] = find_trial_indices(trial_num)

    return trial_vars


def divide_into_trials_dict(trial_series, vars_dict):
    """
    Divide experimental and behavioural variables, indicated as values
    in the dictionary, into their respective trials and put them into
    a Pandas dataframe. Column names are indicated by keys in the
    dictionary.

    Parameters
    ----------
    trial_series :

    vars_dict :


    Returns
    -------
    out :

    """
    # First createp lists to stack the timeseries from different trials
    # Create a second function that can do this so that we could use it
    # for a different set of variables that perhaps cannot be put into
    # a pandas dataframe

    pd_dict = {}

    for key, value in tqdm(vars_dict.items()):

        var_list = []

        for trial in tqdm(np.unique(trial_series)[np.unique(trial_series) >= 0]):

            var_list.append(value[trial_series == trial])

        pd_dict[key] = var_list

    return pd_dict
