"""Levels."""

import os

from configs import config
from configs import samplers
from configs.levels import get_stimuli_dir


def vertical_random_height(**kwargs):
    stimulus_generator = samplers.Sampler(
        stimuli_dir=os.path.join(
            get_stimuli_dir.stimuli_dir(), 'random/VerticalPreyRandomHeight'),
    )
    return config.Config(stimulus_generator, **kwargs)


def vertical_random_height_center(**kwargs):
    stimulus_generator = samplers.Sampler(
        stimuli_dir=os.path.join(
            get_stimuli_dir.stimuli_dir(), 'random/VerticalPreyRandomHeight'),
        filter_fn=lambda f: f['x'] == 3,
    )
    return config.Config(stimulus_generator, **kwargs)
