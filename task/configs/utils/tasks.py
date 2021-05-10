"""Custom tasks."""

from moog import tasks
from dm_env import specs
import numpy as np


class TimeErrorReward(tasks.AbstractTask):
    """Timing task.

    Reward is a tooth function around the time the prey exits the maze in the
    direction of the response.
    """

    def __init__(self,
                 half_width,
                 maximum,
                 prey_speed,
                 response_layer='response',
                 prey_layer='prey'):
        """Constructor.

        Args:
            half_width: reward window (i.e., width of tooth function)
            maximum: maximum reward at zero time error
            prey_speed: time error is computed by dividing distance_remaining with prey_speed
            response_layer: sprite layer
            prey_layer: sprite layer
        """
        self._half_width = half_width
        self._maximum = maximum
        self._prey_speed = prey_speed
        self._response_layer = response_layer
        self._prey_layer = prey_layer

    def reset(self, state, meta_state):
        del state
        del meta_state
        self._reward_given = False

    def _tooth_function(self, speed, distance_remaining):
        time_remaining = distance_remaining / speed
        time_error = np.abs(time_remaining)
        slope = self._maximum / self._half_width
        reward = self._maximum - time_error * slope
        reward = max(reward, 0)
        return reward

    def reward(self, state, meta_state, step_count):
        del step_count
        
        response = [s for s in state['response'] if s.opacity > 0]
        
        if len(response) == 0:
            return 0, False
        else:
            response = response[0]

        if response.x < 0.1:
            response_direction = 0  # left
        elif response.x > 0.9:
            response_direction = 1  # right
        elif response.y < 0.1:
            response_direction = 2  # down
        elif response.y > 0.9:
            response_direction = 3  # up

        prey = state['prey'][0]
        if prey.angle == np.pi / 2:
            prey_direction = 0    # left
        elif prey.angle == -np.pi / 2:
            prey_direction = 1
        elif prey.angle == 0.:  # down
            prey_direction = 2
        elif prey.angle == np.pi:
            prey_direction = 3

        direction_correct = response_direction == prey_direction

        if direction_correct and meta_state['phase'] == 'reward' and not self._reward_given:
            # Update reward
            reward = self._tooth_function(
                self._prey_speed, meta_state['prey_distance_remaining'])
            self._reward_given = True
        else:
            reward = 0

        return reward, False
