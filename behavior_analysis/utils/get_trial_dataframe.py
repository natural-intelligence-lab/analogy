"""Function to load dataframe of trial data and reaction times."""

import numpy as np
import pandas as pd


def get_trial_dataframe(trial_paths, stimulus_features):
    """Create trial dataframe with features for each trial stimulus.

    This dataframe has one row per trial --- i.e. has no information at
    timescales within a trial. If can be used to plot reaction times, trial
    durations, etc.
    """
    trial_df = pd.DataFrame({
        'trial_num': range(len(trial_paths)),
    })

    stim_feature_keys = set()
    for stim_f in stimulus_features:
        stim_feature_keys.update(stim_f.keys())
    stim_feature_keys = list(stim_feature_keys)

    for column_name in stim_feature_keys:
        column = [stim_f.get(column_name, None) for stim_f in stimulus_features]
        trial_df[column_name] = column

    print(f'trial_df columns: {list(trial_df.columns)}')
    print('Unique Values:')
    for k in trial_df.columns:
        if k == 'trial_num':
            continue
        print(f'{k}: {np.unique(trial_df[k])}')

    return trial_df
