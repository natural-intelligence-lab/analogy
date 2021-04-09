"""Levels."""

import os

from configs import config
from configs import samplers
from configs.levels import get_stimuli_dir


def random_4(**kwargs):
    def _filter_fn(f):
        return True
    stimulus_generator = samplers.Sampler(
        stimuli_dir=os.path.join(
            get_stimuli_dir.stimuli_dir(), 'random/Random4_0'),
        filter_fn=_filter_fn,
    )
    return config.Config(stimulus_generator, **kwargs)
