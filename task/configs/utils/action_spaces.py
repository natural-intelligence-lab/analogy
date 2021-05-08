"""Discrete grid action space for controlling agent avatars."""

from moog import action_spaces
from dm_env import specs
import numpy as np


class CardinalDirections(action_spaces.AbstractActionSpace):
    """CardinalDirections action space.

    This action space has 5 actions {left, right, bottom, up, do-nothing}. These
    actions make invisible sprites visible. Only one non-nothing action can be
    taken per trial.
    """

    def __init__(self, action_layer='agent'):
        """Constructor.

        Args:
            action_layer: String. Must be a key in the environment state. There
                must be 4 sprites in that layer, for the 4 directions.

        """
        self._action_layer = action_layer
        self._action_spec = specs.DiscreteArray(5)

    def step(self, state, action):
        """Apply action to environment state.

        Args:
            state: Ordereddict of layers of sprites. Environment state.
            action: Numpy float array of size (2). Force to apply.
        """
        if not self._action_taken:
            if action == 4:
                return
            elif 0 <= action <= 3:
                print(action)
                state[self._action_layer][action].opacity = 255
                self._action_taken = True
            else:
                raise ValueError(f'Invalid action {action}')

    def reset(self, state):
        """Reset action space at start of new episode."""
        del state
        self._action_taken = False

    def random_action(self):
        """Return randomly sampled action."""
        return np.random.randint(5)

    def action_spec(self):
        return self._action_spec
