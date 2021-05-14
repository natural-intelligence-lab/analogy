"""Functions to analyze behavior performance."""

import json
import numpy as np


def get_response_error(trial_path):
    """Get signed response error."""

    trial = json.load(open(trial_path, 'r'))
    step_indices = np.arange(0, len(trial) - 2)
    
    for step in step_indices:
        # 1st two cells are [sprites], {'prey_path','stimulus_features'}
        step_string = trial[step + 2]
        phase = step_string[4][1]['phase']
        if phase == 'reward':
            return step_string[4][1]['prey_distance_remaining']

def get_rt_offline(trial_path):
    """Get RT offline."""
    trial = json.load(open(trial_path, 'r'))
    step_indices = np.arange(0, len(trial) - 2)

    flag_offline = False
    flag_motion_visible = False

    for step in step_indices:
        # 1st two cells are [sprites], {'prey_path','stimulus_features'}
        step_string = trial[step + 2]
        phase = step_string[4][1]['phase']

        if phase == 'offline' and flag_offline == False:
            tOnset = step_string[0][1]
            flag_offline = True
        if phase == 'motion_visible' and flag_motion_visible == False:
            tOffset = step_string[0][1]
            flag_motion_visible = True
            break

    return tOffset-tOnset


def get_image_size(trial_path):
    """Get image size."""

    trial = json.load(open(trial_path, 'r'))
    return trial[2][4][1]['image_size']


def add_response_error_to_trial_df(trial_df, trial_paths):
    response_errors = [
        get_response_error(p) for p in trial_paths
    ]
    trial_df['prey_distance_at_response'] = response_errors

    image_size = [
        get_image_size(p) for p in trial_paths
    ]
    trial_df['image_size'] = image_size

    rt_offline = [
        get_rt_offline(p) for p in trial_paths
    ]
    trial_df['RT_offline'] = rt_offline
