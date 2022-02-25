"""Levels."""

import os


# if getvar('platform') == 'laptop':
# from configs import config_human as config
# elif getvar('platform') == 'desktop':
#     from configs import config_human as config
# elif getvar('platform') == 'psychophysics':
#     from configs import config_human as config
# elif getvar('platform') == 'monkey_ephys':
#     from configs import config as config
# elif getvar('platform') == 'monkey_train':
from configs import config_g as config

from configs import samplers
from configs.levels import get_stimuli_dir

# 2021/9/8
min_num_turns = 2 # 1 # 2021/9/3
max_num_turns = 4 # 1 # 2021/9/3 # 4 # 2 # 2021/8/18
step_num_turns = 2

# 2021/10/9
min_num_turns_human = 0 # 2 # 1 # 2021/9/3
max_num_turns_human = 6 # 4 # 1 # 2021/9/3 # 4 # 2 # 2021/8/18
step_num_turns_human= 2

def random_12(**kwargs):
    stimulus_generator = samplers.Sampler(
        stimuli_dir=os.path.join(
            get_stimuli_dir.stimuli_dir(), 'random/Random12Square'),
    )
    return config.Config(stimulus_generator, **kwargs)


def random_12_uniform_num_turns(**kwargs):
    stim_dir = os.path.join(get_stimuli_dir.stimuli_dir(), 'random/Random12')
    num_turns_samplers = [
        samplers.Sampler(
            stimuli_dir=stim_dir,
            length=20,
            filter_fn=lambda f: f['num_turns'] == i,
        )
        for i in range(6)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers)
    return config.Config(stimulus_generator, **kwargs)

def random_20_uniform_num_turns(**kwargs):
    stim_dir = os.path.join(get_stimuli_dir.stimuli_dir(), 'random/Random20Square')
    num_turns_samplers = [
        samplers.Sampler(
            stimuli_dir=stim_dir,
            length=200,  # 10,
            num_passes=1,  # 20,
            filter_fn=lambda f: f['num_turns'] == i,
        )
        for i in range(min_num_turns_human, max_num_turns_human+1, step_num_turns_human)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers)
    staircase = config.IdTrialStaircase()
    return config.Config(stimulus_generator,id_trial_staircase=staircase, **kwargs)

# 2021/9/8
def random_12_staircase(**kwargs):
    stim_dir = os.path.join(get_stimuli_dir.stimuli_dir(), 'random/Random12Square')
    num_turns_samplers = [
        samplers.Sampler(
            stimuli_dir=stim_dir,
            length=100,
            num_passes=100,
            filter_fn=lambda f: f['num_turns'] == i,
        )
        for i in range(min_num_turns, max_num_turns+1, step_num_turns)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers)
    staircase = config.PreyOpacityStaircase()
    return config.Config(stimulus_generator, prey_opacity_staircase=staircase, **kwargs)


# 2021/12/15
def random_12_staircase_both(**kwargs):
    stim_dir = os.path.join(get_stimuli_dir.stimuli_dir(), 'random/Random12Square')
    num_turns_samplers = [
        samplers.Sampler(
            stimuli_dir=stim_dir,
            length=100,
            num_passes=100,
            filter_fn=lambda f: f['num_turns'] == i,
        )
        for i in range(min_num_turns, max_num_turns+1, step_num_turns)
    ]
    stimulus_generator = samplers.MixtureSampler(*num_turns_samplers)
    staircase = config.PreyOpacityStaircase()
    staircase_path = config.PathPreyOpacityStaircase()
    return config.Config(stimulus_generator, prey_opacity_staircase=staircase, path_prey_opacity_staircase=staircase_path, **kwargs)
