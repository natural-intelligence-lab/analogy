"""Classes to load and sample stimuli."""

import copy
import json
import logging
import numpy as np
import os


class Sampler():
    """Stimulus sampler class.

    This should serve as the `stimulus_generator` argument to a Config().
    """
    
    def __init__(self,
                 stimuli_dir,
                 filter_fn=None,
                 length=np.inf,
                 num_passes=1):
        """Constructor.
        
        Args:
            TODO(nwatters): Add documentation.
        """
        self._length = length
        self._num_passes = num_passes

        if filter_fn is None:
            self._filter_fn = lambda _: True
        else:
            self._filter_fn = filter_fn

        self._load_stimuli(stimuli_dir)

        self._count = 0
        self._pass_num = -1
        self._reset_cycle()

    def _reset_cycle(self):
        """Looped through all stimuli, so re-sampler ordering and loop again."""
        self._cycle = np.random.permutation(len(self._stimuli))
        self._pass_num += 1

    def _process_stimulus_string(self, x):
        """Convert stimulus string from logs into a stimulus for the config."""
        maze_arms, prey_arm, features = x
        if not self._filter_fn(features):
            return []

        stimulus = dict(
            maze_arms=maze_arms,
            prey_arm=prey_arm,
            features=features,
        )
        return stimulus

    def _load_stimuli(self, stimuli_dir):
    
        stimulus_filenames = sorted(
            filter(lambda s: s.isnumeric(), os.listdir(stimuli_dir)))
        stimulus_strings = [
            json.load(open(os.path.join(stimuli_dir, x)))
            for x in stimulus_filenames
        ]

        self._stimuli = []
        for x in stimulus_strings:
            self._stimuli.append(self._process_stimulus_string(x))

        if len(self._stimuli) == 0:
            raise ValueError(
                'No stimuli. Check your filter_fn argument.')

        if self._length < len(self._stimuli):
            self._stimuli = self._stimuli[:self._length]

    def __call__(self):
        """Return stimulus for the config."""
        
        if self._pass_num == self._num_passes:
            # Ran out of stimuli
            return None

        ind = self._cycle[self._count]
        self._count += 1

        if self._count == len(self._stimuli):
            # Finished a cycle through all the stimuli, so begin another cycle
            self._reset_cycle()
            self._count = 0

        stimulus = copy.deepcopy(self._stimuli[ind])

        return stimulus

    def __len__(self):
        return self._num_passes * len(self._stimuli)


class MixtureSampler():

    def __init__(self, *samplers):
        
        self._samplers = samplers
        
        # Create indices for which sampler to use at each trial
        sampler_inds = []
        for i, x in enumerate(samplers):
            sampler_inds.extend(len(x) * [i])
        self._sampler_inds = [
            sampler_inds[i] for i in np.random.permutation(len(sampler_inds))]

        self._count = 0

    def __call__(self):
        if self._count >= len(self._sampler_inds):
            return None
        sampler_ind = self._sampler_inds[self._count]
        self._count += 1
        return self._samplers[sampler_ind]()

    def __len__(self):
        return len(self._sampler_inds)


class ChainedSampler():

    def __init__(self, *samplers):
        self._samplers = samplers
        self._sampler_ind = 0

    def __call__(self):
        sampler = self._samplers[self._sampler_ind]
        stimulus = sampler()
        if stimulus is None:
            self._sampler_ind += 1
            if self._sampler_ind >= len(self._samplers):
                return None
            else:
                return self()
        else:
            return stimulus

    def __len__(self):
        return sum([len(x) for x in self._samplers])
