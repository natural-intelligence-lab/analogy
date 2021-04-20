"""Discrete grid action space for controlling agent avatars."""

from moog import action_spaces
from dm_env import specs
import numpy as np


class GridRotate(action_spaces.AbstractActionSpace):
    """Discrete grid action space.

    This action space has 3 actions {left turn, right turn, do-nothing}. These
    actions control either the angular velocity of the agent(s). Put zero matrix
    for consistency with Grid action spaces (move up/down)
    """

    _ACTIONS = (
        np.array(np.pi/2),  # turn left
        np.array(-np.pi/2),  # turn right
        np.array(0),  # Do not move
        np.array(0),  # Do not move
        np.array(0),  # Do not move
    )

    def __init__(self, scaling_factor=1, action_layers='agent'):
        """Constructor.

        Args:
            scaling_factor: Scalar. Scaling factor multiplied to the action.
                default 1
            agent_layer: String or iterable of strings. Elements (or itself if
                string) must be keys in the environment state. All sprites in
                these layers will be acted upon by this action space.

        """
        self._scaling_factor = scaling_factor
        if not isinstance(action_layers, (list, tuple)):
            action_layers = (action_layers,)
        self._action_layers = action_layers

        self._action_spec = specs.DiscreteArray(len(self._ACTIONS))

    def step(self, state, action):
        """Apply action to environment state.

        Args:
            state: Ordereddict of layers of sprites. Environment state.
            action: Numpy float array of size (2). Force to apply.
        """
        self._action *= 0

        self._action += GridRotate._ACTIONS[action]*self._scaling_factor

        for action_layer in self._action_layers:
            for sprite in state[action_layer]:
                sprite.angle += self._action[0]

                # sprite.angle_vel += self._action[0]

    def reset(self, state):
        """Reset action space at start of new episode."""
        del state
        self._action = np.zeros(2)

    def random_action(self):
        """Return randomly sampled action."""
        return np.random.randint(len(GridRotate._ACTIONS))

    def action_spec(self):
        return self._action_spec
