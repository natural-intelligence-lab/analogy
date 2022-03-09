"""Levels."""

import os
# from configs import config_human as config
from configs import config
from configs import samplers
from configs.levels import get_stimuli_dir

_min_num_turns = 2
_max_num_turns = 6 # 4
_step_num_turns = 2
_num_layers= 2 # 50

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

def random_16_staircase(num_layers=_num_layers, max_num_turns=_max_num_turns,**kwargs):
    """Random layered mazes within a grid of size 16."""
    path_dir = os.path.join(
        get_stimuli_dir.stimuli_dir(),
        'wire_mazes/maze_size_16/samples_per_pair_100_v0',
    )
    num_turns_samplers = [samplers.WireMazeSampler(
            path_dir=path_dir,
            num_layers=num_layers,
            ball_path_top_bottom=True,
            distractors_top_bottom=True,
            max_num_turns=max_num_turns,
        )
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers,
        num_passes=100)
    staircase = config.PreyOpacityStaircase()
    staircase_path = config.PathPreyOpacityStaircase()
    return config.Config(stimulus_generator, prey_opacity_staircase=staircase, path_prey_opacity_staircase=staircase_path, **kwargs)
