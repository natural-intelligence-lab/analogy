"""Function to get trial paths from a dataset path."""

import json
import numpy as np
import os

# This is the index of meta_state in each timestep's
_META_STATE_LOG_INDEX = 4


def _get_stimulus_features(trial_path):
    trial = json.load(open(trial_path))
    timestep_0 = trial[2]  # First two entries are full state and maze matrix
    stim_features = timestep_0[_META_STATE_LOG_INDEX][1]['stimulus_features']
    return stim_features


def get_trial_paths(data_path):

    # Chronological trial paths
    trial_paths = [
        os.path.join(data_path, x)
        for x in sorted(os.listdir(data_path)) if x.isnumeric()
    ]

    # Stimulus features
    stimulus_features = [_get_stimulus_features(x) for x in trial_paths]

    # Print number of trials and unique stimulus names
    num_trials = len(trial_paths)
    print(f'Number of trials:  {num_trials}')

    return trial_paths, stimulus_features
