"""Functions to analyze behavior performance."""

import json
import numpy as np


def get_response_error(trial_path):
    """Get signed response error."""

    trial = json.load(open(trial_path, 'r'))
    step_indices = np.arange(0, len(trial) - 2)
    
    for step in step_indices:
        step_string = trial[step + 2]
        phase = step_string[4][1]['phase']
        if phase == 'reward':
            return step_string[4][1]['prey_distance_remaining']


def add_response_error_to_trial_df(trial_df, trial_paths):
    response_errors = [
        get_response_error(p) for p in trial_paths
    ]
    trial_df['prey_distance_at_response'] = response_errors
