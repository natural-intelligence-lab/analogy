"""Custom tasks."""

from moog import tasks
from dm_env import specs
import numpy as np


class TimeErrorReward(tasks.AbstractTask):
    """Timing task.

    Reward is a tooth function around the time the prey reaches the agent's
    position.
    """

    def __init__(self,
                 half_width,
                 maximum,
                 prey_speed,
                 agent_layer='agent',
                 prey_layer='prey'):
        """Constructor.

        Args:
            half_width: reward window (i.e., width of tooth function)
            maximum: maximum reward at zero time error
            prey_speed: time error is computed by dividing distance_remaining with prey_speed
            agent_layer: sprite layer
            prey_layer: sprite layer
        """
        self._half_width = half_width
        self._maximum = maximum
        self._prey_speed = prey_speed
        self._agent_layer = agent_layer
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
        prey = state['prey'][0]
        agent = state['agent'][0]
        direction_correct = False

        # pacman facing left and prey from left or
        # pacman facing right and prey from right
        # sprite.angle = -np.pi/2*d[1]  # for reward; d[1]=1,-1 for (r,l); 12 o'clock = degree zero & CCW for +
        if (agent.angle==np.pi/2 and prey.angle==-np.pi/2) or (agent.angle == -np.pi / 2 and prey.angle==np.pi/2):
            direction_correct = True

        del state
        del step_count

        if direction_correct and meta_state['phase'] == 'reward' and not self._reward_given:
            # Update reward
            reward = self._tooth_function(
                self._prey_speed, meta_state['prey_distance_remaining'])
            self._reward_given = True
        else:
            reward = 0

        return reward, False
