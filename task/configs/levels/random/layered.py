"""Levels."""

import os
from configs import config
# from configs import config_human as config
from configs import samplers
from configs.levels import get_stimuli_dir
import numpy as np

# 2022/1/31
_min_num_turns = 1 # 2
_max_num_turns = 4
_step_num_turns = 2 # 1 # 2
_num_layers=50 # 7  # 2022/2/17

min_num_turns_human = 2
max_num_turns_human = 6
step_num_turns_human= 2
_num_layers_human=15


def random_12(num_layers=5, max_num_turns=4, **kwargs):
    """Random layered mazes within a grid of size 12."""
    path_dir = os.path.join(
        get_stimuli_dir.stimuli_dir(),
        'layered_mazes/maze_size_12/samples_per_pair_100_v0',
    )
    stimulus_generator = samplers.LayeredMazeSampler(
        path_dir=path_dir,
        num_layers=num_layers,
        max_num_turns=max_num_turns,
    )
    return config.Config(stimulus_generator, **kwargs)


def random_14(num_layers=5, max_num_turns=4, **kwargs):
    """Random layered mazes within a grid of size 14."""
    path_dir = os.path.join(
        get_stimuli_dir.stimuli_dir(),
        'layered_mazes/maze_size_14/samples_per_pair_100_v0',
    )
    stimulus_generator = samplers.LayeredMazeSampler(
        path_dir=path_dir,
        num_layers=num_layers,
        max_num_turns=max_num_turns,
    )
    return config.Config(stimulus_generator, **kwargs)


def random_14_staircase(num_layers=_num_layers, max_num_turns=_max_num_turns,**kwargs):
    """Random layered mazes within a grid of size 14."""
    path_dir = os.path.join(
        get_stimuli_dir.stimuli_dir(),
        'layered_mazes/maze_size_14/samples_per_pair_100_v0',
    )
    num_turns_samplers = [
        samplers.LayeredMazeSampler(
            path_dir=path_dir,
            num_layers=num_layers,
            ball_path_top_bottom=False,
            ball_path_top=True,
            max_num_turns=max_num_turns,
            num_turns=i,
        )
        for i in range(_min_num_turns, max_num_turns+1, _step_num_turns)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers,
        num_passes=100)
    staircase = config.PreyOpacityStaircase()
    staircase_path = config.PathPreyOpacityStaircase()
    return config.Config(stimulus_generator, prey_opacity_staircase=staircase, path_prey_opacity_staircase=staircase_path, **kwargs)

def random_20(num_layers=15, max_num_turns=4, **kwargs):
    """Random layered mazes within a grid of size 20."""
    path_dir = os.path.join(
        get_stimuli_dir.stimuli_dir(),
        'layered_mazes/maze_size_20/samples_per_pair_50_v0',
    )
    stimulus_generator = samplers.LayeredMazeSampler(
        path_dir=path_dir,
        num_layers=num_layers,
        max_num_turns=max_num_turns,
    )
    return config.Config(stimulus_generator, **kwargs)

def random_20_uniform_num_turns(num_layers=_num_layers_human,**kwargs):
    """Random layered mazes within a grid of size 20."""
    path_dir = os.path.join(
        get_stimuli_dir.stimuli_dir(),
        'layered_mazes/maze_size_20/samples_per_pair_50_v0',
    )
    num_turns_samplers = [
        samplers.LayeredMazeSampler(
            path_dir=path_dir,
            num_layers=num_layers,
            max_num_turns=max_num_turns_human,
            num_turns=i,
        )
        for i in range(min_num_turns_human, max_num_turns_human+1, step_num_turns_human)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers)
    staircase = config.IdTrialStaircase()
    return config.Config(stimulus_generator,id_trial_staircase=staircase, **kwargs)