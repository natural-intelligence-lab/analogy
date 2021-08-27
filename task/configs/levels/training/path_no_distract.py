"""Levels."""

import os

from configs import config
from configs import samplers
from configs.levels import get_stimuli_dir

min_num_turns = 0
max_num_turns = 4 # 2 # 2021/8/18
step_num_turns = 2

max_num_turns_g = 2 # for G, 2021/8/27

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