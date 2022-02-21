"""Levels."""

import os
import numpy as np

from configs import config
from configs import samplers
from configs.levels import get_stimuli_dir

min_num_turns = 0 # 1 # 2021/9/3
max_num_turns = 2 # 10/21 # 0 # 2021/10/4 # 4 # 1 # 2021/9/3 # 4 # 2 # 2021/8/18
step_num_turns = 2

max_num_turns_g = 2 # for G, 2021/8/27

_NUM_GRID_START_X = 6  # minimum difference of start_x and start_x_distract

def path_no_distract_uniform_num_turns(**kwargs):
    stim_dir = os.path.join(get_stimuli_dir.stimuli_dir(), 'random/PathNoDistract')
    num_turns_samplers = [
        samplers.Sampler(
            stimuli_dir=stim_dir,
            length=100,
            filter_fn=lambda f: f['num_turns'] == i,
        )
        for i in range(min_num_turns, max_num_turns+1, step_num_turns)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers)
    return config.Config(stimulus_generator, **kwargs)

def path_no_distract_uniform_num_turns_staircase(**kwargs):
    stim_dir = os.path.join(get_stimuli_dir.stimuli_dir(), 'random/PathNoDistract')
    num_turns_samplers = [
        samplers.Sampler(
            stimuli_dir=stim_dir,
            length=100,
            num_passes=100,
            filter_fn=lambda f: f['num_turns'] == i,
        )
        for i in range(min_num_turns, max_num_turns_g+1, step_num_turns)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers)
    staircase = config.PreyOpacityStaircase()
    return config.Config(stimulus_generator, prey_opacity_staircase=staircase, **kwargs)

def path_no_distract_even_odd_num_turns_staircase(**kwargs):
    stim_dir = os.path.join(get_stimuli_dir.stimuli_dir(), 'random/PathNoDistract')
    num_turns_samplers = [
        samplers.Sampler(
            stimuli_dir=stim_dir,
            length=100,
            num_passes=100,
            filter_fn=lambda f: f['num_turns'] == i,
        )
        for i in range(min_num_turns, max_num_turns+1, 1) # step_num_turns)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers)
    staircase = config.PreyOpacityStaircase()
    return config.Config(stimulus_generator, prey_opacity_staircase=staircase, **kwargs)

def path_no_distract(**kwargs):
    stimulus_generator = samplers.Sampler(
        stimuli_dir=os.path.join(
            get_stimuli_dir.stimuli_dir(), 'random/PathNoDistract'),
    )
    return config.Config(stimulus_generator, **kwargs)

def path_partial_distract_even_odd_num_turns_staircase(**kwargs):
    stim_dir = os.path.join(get_stimuli_dir.stimuli_dir(), 'random/PathPartialDistract')
    num_turns_samplers = [
        samplers.Sampler(
            stimuli_dir=stim_dir,
            length=100,
            num_passes=100,
            filter_fn=lambda f: f['num_turns'] == i,            
        )
        for i in range(min_num_turns, max_num_turns+1, 1) # step_num_turns)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers)
    staircase = config.PreyOpacityStaircase()
    return config.Config(stimulus_generator, prey_opacity_staircase=staircase, **kwargs)


# training.path_no_distract.path_distract_path_even_odd_num_turns_staircase", fixation_phase=False, prey_opacity=0, ms_per_unit=3000)') // 4000

def path_distract_path_even_odd_num_turns_staircase(**kwargs):
    stim_dir = os.path.join(get_stimuli_dir.stimuli_dir(), 'random/PathDistractPath')
    num_turns_samplers = [
        samplers.Sampler(
            stimuli_dir=stim_dir,
            length=100,
            num_passes=100,
            filter_fn=lambda f: (f['num_turns'] == i and f['num_turns_distract'] == (max_num_turns)-i),
            # and (np.abs(f['start_x']-f['start_x_distract'])>_NUM_GRID_START_X)),
        )
        for i in range(min_num_turns, max_num_turns+1, step_num_turns) # 1) # step_num_turns)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers)
    staircase = config.PreyOpacityStaircase()
    return config.Config(stimulus_generator, prey_opacity_staircase=staircase, **kwargs)
