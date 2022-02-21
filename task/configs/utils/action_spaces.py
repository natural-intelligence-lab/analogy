"""Discrete grid action space for controlling response avatars."""

from moog import action_spaces
from dm_env import specs
import numpy as np


class CardinalDirections(action_spaces.AbstractActionSpace):
    """CardinalDirections action space.

    This action space has 5 actions {left, right, bottom, up, do-nothing}. These
    actions make invisible sprites visible. Only one non-nothing action can be
    taken per trial.
    """

    def __init__(self, action_layer='response'):
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


class JoystickColor(action_spaces.AbstractActionSpace):
    """JoystickColor action space.
    
    Moves agent left/right and changes color if joystick is pressed up.
    """

    DIRECTIONS = [
        np.array([0, 1]),  # Up
        np.array([0, -1]),  # Down
        np.array([1, 0]),  # Right
        np.array([-1, 0]),  # Left
        np.array([0, 0]),  # Nothing
    ]

    def __init__(self, up_color, scaling_factor=1., action_layer='agent',
                 joystick_layer='joystick'):
        """Constructor.
        
        Args:
            up_color: Tuple. Color upon up motion.
            scaling_factor: Scalar. Scaling factor multiplied to the action.
            agent_layer: String or iterable of strings. Elements (or itself if
                string) must be keys in the environment state. All sprites in
                these layers will be acted upon by this action space.
        """
        self._up_color =  up_color
        self._scaling_factor = scaling_factor
        self._action_layer = action_layer
        self._joystick_layer = joystick_layer

        self._action_spec = specs.BoundedArray(
            shape=(2,), dtype=np.float32, minimum=-1, maximum=1)

    def step(self, state, action):
        """Apply action to environment state.

        Args:
            state: OrderedDict. Environment state.
            action: Numpy float array of size (2) in [-1, 1]. Force to apply.
        """

        # Move joystick sprite  # for joystick fixation
        for sprite in state[self._joystick_layer]:
            sprite.position = 0.4 * action + 0.5

        # Project to cardinal directions
        if np.all(action == 0): # no action
            action_index = 4
        else: # max a direction
            dots = [np.dot(action, d) for d in JoystickColor.DIRECTIONS]
            action_index = np.argmax(dots)
        action = self._scaling_factor * np.linalg.norm(action) * (  # non-max direction also contribute to max direction
            JoystickColor.DIRECTIONS[action_index])

        # Move the sprite
        for sprite in state[self._action_layer]:
            # agent glued (set mass to inf) from visible motion phase
            if action_index == 0 and not sprite.metadata['response_up'] and not np.isfinite(sprite.mass): # up
                sprite.c0 = self._up_color[0]
                sprite.c1 = self._up_color[1]
                sprite.c2 = self._up_color[2]
                sprite.metadata['response_up'] = True

            if np.isfinite(sprite.mass):
                if action_index in (2, 3): # left/right
                    sprite.velocity = action / sprite.mass
                else: # when max direction is either up/down
                    sprite.velocity = action / sprite.mass
                    # sprite.velocity = np.zeros(2) # action / sprite.mass
                    sprite.metadata['y_speed'] = action[1] / sprite.mass
            else: # glued
                sprite.velocity = np.zeros(2)

    def random_action(self):
        """Return randomly sampled action."""
        return np.random.uniform(-1., 1., size=(2,))
    
    def action_spec(self):
        return self._action_spec
