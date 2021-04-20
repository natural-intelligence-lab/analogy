"""Constants."""

import collections
import numpy as np


class ArmLengthsClass():

    def __init__(self, arm_lengths):
        self._arm_lengths = arm_lengths

    def __getitem__(self, i):
        """Get arm length."""
        return self._arm_lengths[i]

    def __len__(self):
        return len(self._arm_lengths)

    def __iter__(self):
        return iter(self._arm_lengths)

    def sample(self, p=None):
        return np.random.choice(self._arm_lengths, p=p)

    @property
    def arm_lengths(self):
        return self._arm_lengths

ArmLengths = ArmLengthsClass(arm_lengths=(3, 4, 5, 6))


DirectionsClass = collections.namedtuple('Directions',['u', 'd', 'l', 'r'])

Directions = DirectionsClass(
    u=np.array([0., 1.]),
    d=np.array([0., -1.]),
    r=np.array([1., 0.]),
    l=np.array([-1., 0.]),
)

max_reward = 100  # in unit of [ms]
reward_window = 0.15  # in unit of [(tp-ts)/ts]
bonus_reward = 0  # in unit of [ms]