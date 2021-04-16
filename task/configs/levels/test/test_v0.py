"""Levels."""

import os

from configs import config
from configs import samplers
from configs.levels import get_stimuli_dir


def test_v0(**kwargs):
    stimulus_generator = samplers.Sampler(
        stimuli_dir=os.path.join(
            get_stimuli_dir.stimuli_dir(), 'test/TestV0'),
        num_passes=3,
    )
    return config.Config(stimulus_generator, **kwargs)
