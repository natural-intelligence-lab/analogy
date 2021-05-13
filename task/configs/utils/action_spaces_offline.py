"""Discrete grid action space for controlling response avatars."""

from moog import action_spaces
from dm_env import specs
import numpy as np


class YesNoResponse(action_spaces.AbstractActionSpace):
    """Binary (yes/no) action space.

    This action space has 2 actions {yes, no/do-nothing}. These
    actions make invisible sprites visible. Only one non-nothing action can be
    taken per trial.
    """

    def __init__(self, action_layer='responses_offline'):
        """Constructor.

        Args:
            action_layer: String. Must be a key in the environment state. There
                must be 1 sprite in that layer, for the yes/no response.

        """
        self._action_layer = action_layer
        self._action_spec = specs.DiscreteArray(2)

    def step(self, state, action):
        """Apply action to environment state.

        Args:
            state: Ordereddict of layers of sprites. Environment state.
            action: Numpy float array of size (2). Force to apply.
        """
        if not self._action_taken:
            if action == 0:
                for i in range(len(state[self._action_layer])):
                    state[self._action_layer][i].opacity = 128
                self._action_taken = True
            elif action == 1:
                return
            else:
                raise ValueError(f'Invalid action {action}')

    def reset(self, state):
        """Reset action space at start of new episode."""
        del state
        self._action_taken = False

    def random_action(self):
        """Return randomly sampled action."""
        return np.random.randint(2)

    def action_spec(self):
        return self._action_spec
