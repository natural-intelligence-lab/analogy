"""Custom tasks."""

from moog import tasks
from dm_env import specs
import numpy as np
import inspect

_EPSILON = 1e-4


class TimeErrorReward(tasks.AbstractTask):
    """Timing task.

    Reward is a tooth function around the time the prey exits the maze in the
    direction of the response.
    """

    def __init__(self,
                 half_width,
                 maximum,
                 prey_speed,
                 max_rewarding_dist,
                 prey_opacity_staircase,
                 response_layer='agent',
                 prey_layer='prey'):
        """Constructor.

        Args:
            half_width: reward window (i.e., width of tooth function)
            maximum: maximum reward at zero time error
            prey_speed: time error is computed by dividing distance_remaining with prey_speed
            max_rewarding_dist: Scalar. Maximum distance (in units of screen
                width) from the correct exit to give reward.
            response_layer: sprite layer
            prey_layer: sprite layer

        """
        self._half_width = half_width
        self._maximum = maximum
        self._prey_speed = prey_speed
        self._max_rewarding_dist = max_rewarding_dist
        self._response_layer = response_layer
        self._prey_layer = prey_layer
        self._prey_opacity_staircase = prey_opacity_staircase

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

        if meta_state['phase'] == 'reward' and not self._reward_given:  # and state['agent'][0].metadata['response']
            # Update reward

            agent = state['agent'][0]
            prey_exit_x = meta_state['prey_path'][-1][0]
            
            if np.abs(agent.x - prey_exit_x) < self._max_rewarding_dist:
                reward = self._tooth_function(
                    self._prey_speed, meta_state['prey_distance_remaining'])
            else:
                # Agent is too far away from prey exit in the x axis
                reward = 0
            self._reward_given = True

            if self._prey_opacity_staircase is not None:
                self._prey_opacity_staircase.step(reward)
        else:
            reward = 0

        return reward, False




class OfflineReward(tasks.AbstractTask):
    """Give reward if agent stops at correct exit during offline phase."""

    def __init__(self, phase, max_rewarding_dist=0.):
        """Constructor.
        
        Args:
            phase: String. Phase of task in which to give reward.
            max_rewarding_dist: Scalar. Maximum distance (in units of screen
                width) from the correct exit to give reward. The reward is
                linearly interpolated between zero at this value and 1 at the
                correct exit.
        """
        self._phase = phase
        self._max_rewarding_dist = max_rewarding_dist

    def reset(self, state, meta_state):
        del state
        del meta_state
        self._reward_given = False

    def reward(self, state, meta_state, step_count):
        del step_count
        if len(state['agent']) > 0:
            agent = state['agent'][0]
            if (meta_state['phase'] == self._phase and
                    not self._reward_given and
                    agent.metadata['moved_h'] and
                    np.all(state['agent'][0].velocity == 0)):
                prey_exit_x = meta_state['prey_path'][-1][0]
                agent_prey_dist = np.abs(agent.x - prey_exit_x)
                reward = max(0, 1 - agent_prey_dist / (self._max_rewarding_dist + _EPSILON))
                self._reward_given = True
            else:
                reward = 0.
        else:
            reward = 0.
        
        return reward, False


class BeginPhase(tasks.AbstractTask):
    """Task to give reward at beginning of phase."""

    def __init__(self, phase):
        """Constructor."""
        self._phase = phase

    def reset(self, state, meta_state):
        del state
        del meta_state
        self._reward_given = False

    def reward(self, state, meta_state, step_count):
        del state
        del step_count

        if meta_state['phase'] == self._phase and not self._reward_given:
            self._reward_given = True
            return 0, False
        else:
            return 0, False