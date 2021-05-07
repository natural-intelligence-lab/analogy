"""Levels."""

import os

from configs import config
from configs import samplers
from configs.levels import get_stimuli_dir


def random_12(**kwargs):
    stimulus_generator = samplers.Sampler(
        stimuli_dir=os.path.join(
            get_stimuli_dir.stimuli_dir(), 'random/Random12'),
        filter_fn=lambda f: f['num_turns'] == 1,
    )
    return config.Config(stimulus_generator, **kwargs)
