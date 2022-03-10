"""Levels."""

import os
# from configs import config_human as config
from configs import config
from configs import samplers
from configs.levels import get_stimuli_dir

_MIN_NUM_TURNS = 2
_MAX_NUM_TURNS = 6 # 4
_STEP_NUM_TURNS = 2
_NUM_LAYERS= 2 # 50

_MIN_NUM_OVERLAP=1
_MIN_EXIT_DISTANCE = 3  # GRID; if set to zero, no constraint on the exit distance


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

def random_16_staircase(**kwargs):
    """Random layered mazes within a grid of size 16."""
    path_dir = os.path.join(
        get_stimuli_dir.stimuli_dir(),
        'wire_mazes/maze_size_16/samples_per_pair_100_v0',
    )
    num_turns_samplers = [samplers.WireMazeSampler(
            path_dir=path_dir,
            num_layers=_NUM_LAYERS,
            ball_path_top_bottom=True,
            distractors_top_bottom=True,
            max_num_turns=_MAX_NUM_TURNS,
            min_num_overlap=_MIN_NUM_OVERLAP,
            min_exit_distance=_MIN_EXIT_DISTANCE,
        )
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers,
        num_passes=100)
    staircase = config.PreyOpacityStaircase()
    staircase_path = config.PathPreyOpacityStaircase()
    return config.Config(stimulus_generator, prey_opacity_staircase=staircase, path_prey_opacity_staircase=staircase_path, **kwargs)
