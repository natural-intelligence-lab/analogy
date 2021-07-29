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

def _get_segment_length(trial_path):
    trial = json.load(open(trial_path))
    maze_matrix = trial[1]  # W, H, prey_path,stimulus_features
    prey_path = maze_matrix['prey_path']
    path_length = maze_matrix['stimulus_features']['path_length']
    num_turns = maze_matrix['stimulus_features']['num_turns']
    distance_matrix = np.diff(prey_path, n=1, axis=0)

    segment_length= [0 for i in range(num_turns+1)]
    segment_length[0] = np.sum(np.abs(distance_matrix[0]))
    cell_size=np.sum(np.abs(distance_matrix[0]))
    segment_index = 0
    for i in range(np.size(distance_matrix,0)-1):
        if np.sum(np.abs(distance_matrix[i]- distance_matrix[i+1])) < (cell_size/2):
            segment_length[segment_index] += np.sum(np.abs(distance_matrix[i+1]))
        else:
            segment_index += 1
            segment_length[segment_index] = np.sum(np.abs(distance_matrix[i+1]))

    segment_length = segment_length/cell_size
    return segment_length

def _get_prey_path(trial_path):
    trial = json.load(open(trial_path))
    maze_matrix = trial[1]  # W, H, prey_path,stimulus_features
    prey_path = maze_matrix['prey_path']
    return prey_path

def get_trial_paths(data_path):

    # Chronological trial paths
    trial_paths = [
        os.path.join(data_path, x)
        for x in sorted(os.listdir(data_path)) if x.isnumeric()
    ]

    # Stimulus features
    stimulus_features = [_get_stimulus_features(x) for x in trial_paths]

    # segment_length
    segment_length = [_get_segment_length(x) for x in trial_paths]

    # prey path
    prey_path = [_get_prey_path(x) for x in trial_paths]

    # Print number of trials and unique stimulus names
    num_trials = len(trial_paths)
    print(f'Number of trials:  {num_trials}')

    return trial_paths, stimulus_features, segment_length, prey_path
