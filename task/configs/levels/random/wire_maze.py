"""Levels."""

import os
# from configs import config_human as config
from configs import config
from configs import samplers
from configs.levels import get_stimuli_dir
import numpy as np

# for 16 grid
_MIN_NUM_TURNS = 0 #    2 # 0 # 2 # inclusive
_MAX_NUM_TURNS = 3 # 1 # 3 # 5 # np.inf # 0 # 2 # 6 # 4 # exclusive
# _STEP_NUM_TURNS = 2
_NUM_LAYERS= 2 # 3 # 5 # 4 # 3 # 2 # 50


_MIN_NUM_OVERLAP= 1 # 0 # 1
_MAX_NUM_OVERLAP= np.inf # 2 # 1 # np.inf # 2 # 1 # np.inf
_MIN_EXIT_DISTANCE = 12 # 3 # 0.7/16*3>0.1 # 1 # 0 # 2 # 3  # GRID; if set to zero, no constraint on the exit distance

_DISTRACTOR_NUMBER_TURNS=[0,2]  # [0]  # [0,2]

# ## for 6 grid
# _MIN_NUM_TURNS = 0 # 2 # 0 # 2 # inclusive
# _MAX_NUM_TURNS = 5 # np.inf # 0 # 2 # 6 # 4 # exclusive
# # _STEP_NUM_TURNS = 2
# _NUM_LAYERS= 3 # 5 # 4 # 3 # 2 # 50
#
# _MIN_NUM_OVERLAP= 0 # 1
# _MAX_NUM_OVERLAP= np.inf # 2 # 1 # np.inf
# _MIN_EXIT_DISTANCE = 0

def random_6_staircase(**kwargs):
    """now fix exits equally divided at bottom """
    path_dir = os.path.join(
        get_stimuli_dir.stimuli_dir(),
        # 'wire_mazes/maze_size_6/samples_per_pair_200_v1',  # min seg length of 1
        'wire_mazes/maze_size_6/samples_per_pair_100_v0',  # min seg length of 2
    )
    num_turns_samplers = [
        samplers.WireMazeSampler(
            path_dir=path_dir,
            num_layers=_NUM_LAYERS,
            ball_path_top_bottom=True,
            distractors_top_bottom=True,
            min_num_turns= _MIN_NUM_TURNS,
            max_num_turns=_MAX_NUM_TURNS,
            min_num_overlap=_MIN_NUM_OVERLAP,
            max_num_overlap=_MAX_NUM_OVERLAP,
            min_exit_distance=_MIN_EXIT_DISTANCE, # -np.inf,
        )
        # for i in range(_min_num_turns, max_num_turns + 1, _step_num_turns)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers,
        num_passes=100)
    staircase = config.PreyOpacityStaircase()
    staircase_path = config.PathPreyOpacityStaircase()
    staircase_path_position = config.PathPreyPositionStaircase()
    update_p_correct = config.UpdatePercentCorrect()
    repeat_incorrect_trial = config.RepeatIncorrectTrial()
    return config.Config(stimulus_generator,
                         prey_opacity_staircase=staircase,
                         path_prey_opacity_staircase=staircase_path,
                         path_prey_position_staircase=staircase_path_position,
                         update_p_correct=update_p_correct,
                         repeat_incorrect_trial=repeat_incorrect_trial,
                         **kwargs)

def random_16_staircase(**kwargs):
    """Random layered mazes within a grid of size 16."""
    path_dir = os.path.join(
        get_stimuli_dir.stimuli_dir(),
        # 'wire_mazes/maze_size_6/samples_per_pair_200_v1',  # min seg length of 1
        'wire_mazes/maze_size_16/samples_per_pair_100_v0',  # min seg length of 2
    )
    num_turns_samplers = [
        samplers.WireMazeSampler(
            path_dir=path_dir,
            num_layers=_NUM_LAYERS,
            ball_path_top_bottom=True,
            distractors_top_bottom=True,
            min_num_turns= _MIN_NUM_TURNS,
            max_num_turns=_MAX_NUM_TURNS,
            min_num_overlap=_MIN_NUM_OVERLAP,
            max_num_overlap=_MAX_NUM_OVERLAP,
            min_exit_distance=_MIN_EXIT_DISTANCE, # -np.inf,
            distractors_num_turns=_DISTRACTOR_NUMBER_TURNS,
        )
        # for i in range(_min_num_turns, max_num_turns + 1, _step_num_turns)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers,
        num_passes=100)
    staircase = config.PreyOpacityStaircase()
    staircase_path = config.PathPreyOpacityStaircase()
    staircase_path_position = config.PathPreyPositionStaircase()
    update_p_correct = config.UpdatePercentCorrect()
    repeat_incorrect_trial = config.RepeatIncorrectTrial()
    return config.Config(stimulus_generator,
                         prey_opacity_staircase=staircase,
                         path_prey_opacity_staircase=staircase_path,
                         path_prey_position_staircase=staircase_path_position,
                         update_p_correct=update_p_correct,
                         repeat_incorrect_trial=repeat_incorrect_trial,
                         **kwargs)


def random_16(num_layers=4, max_num_turns=6, **kwargs):
    """Random layered mazes within a grid of size 16."""
    path_dir = os.path.join(
        get_stimuli_dir.stimuli_dir(),
        'wire_mazes/maze_size_16/samples_per_pair_100_v0',
    )
    stimulus_generator = samplers.WireMazeSampler(
        path_dir=path_dir,
        num_layers=num_layers,
        max_num_turns=max_num_turns,
    )
    return config.Config(stimulus_generator, **kwargs)