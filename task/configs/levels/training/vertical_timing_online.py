"""Levels."""

import os

from configs import config_online as config
from configs import samplers
from configs.levels import get_stimuli_dir


def vertical_timing_center(**kwargs):
    stimulus_generator = samplers.Sampler(
        stimuli_dir=os.path.join(
            get_stimuli_dir.stimuli_dir(), 'training/VerticalTiming'),
        filter_fn=lambda f: f['x'] == 5,
        num_passes=int(1e3),
    )
    return config.Config(stimulus_generator, **kwargs)


def vertical_timing_random_x(**kwargs):
    stimulus_generator = samplers.Sampler(
        stimuli_dir=os.path.join(
            get_stimuli_dir.stimuli_dir(), 'training/VerticalTiming'),
        filter_fn=lambda f: f['x'] != 5,
        num_passes=int(1e2),
    )
    return config.Config(stimulus_generator, **kwargs)
