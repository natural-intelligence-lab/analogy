"""Levels."""

import os

from configs import config
from configs import samplers
from configs.levels import get_stimuli_dir


def random_3(**kwargs):
    stimulus_generator = samplers.Sampler(
        stimuli_dir=os.path.join(
            get_stimuli_dir.stimuli_dir(), 'random/Random3'),
        num_passes=3,
        # filter_fn=lambda f: f['prey_arm_turns'] == 3,
    )
    return config.Config(stimulus_generator, **kwargs)
