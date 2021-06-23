"""Levels."""

import os

# from configs import config
from configs import config_human as config
from configs import samplers
from configs.levels import get_stimuli_dir


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
