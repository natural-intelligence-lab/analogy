"""Custom tasks."""

from moog import tasks
from dm_env import specs
import numpy as np
import inspect


class TimeErrorReward(tasks.AbstractTask):
    """Timing task.

    Reward is a tooth function around the time the prey exits the maze in the
    direction of the response.
    """

    def __init__(self,
                 half_width,
                 maximum,
                 prey_speed,
                 response_layer='agent',
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

    def _get_x_distance_threshold(self, agent, prey):
        """Get maximum x displacement for agent and prey to still intersect."""
        agent_x_vertices = agent.vertices[:, 0]
        agent_width = np.max(agent_x_vertices) - np.min(agent_x_vertices)
        prey_x_vertices = prey.vertices[:, 0]
        prey_width = np.max(prey_x_vertices) - np.min(prey_x_vertices)
        x_distance_threshold = 0.5 * (agent_width + prey_width)
        return x_distance_threshold

    def reward(self, state, meta_state, step_count):
        del step_count

        if meta_state['phase'] == 'reward' and not self._reward_given:  # and state['agent'][0].metadata['response']
            # Update reward

            agent = state['agent'][0]
            prey = state['prey'][0]
            x_distance_threshold = self._get_x_distance_threshold(agent, prey)
            prey_exit_x = meta_state['prey_path'][-1][0]
            
            if np.abs(agent.x - prey_exit_x) < x_distance_threshold:
                reward = self._tooth_function(
                    self._prey_speed, meta_state['prey_distance_remaining'])
            else:
                # Agent is too far away from prey exit in the x axis
                reward = 0
            self._reward_given = True
        else:
            reward = 0

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

class OfflineReward(tasks.AbstractTask):
    """Task to give reward if agent move to correct exit."""

    def __init__(self, phase):
        """Constructor."""
        self._phase = phase

    def reset(self, state, meta_state):
        del state
        del meta_state
        self._reward_given = False

    def _get_x_distance_threshold(self, agent, prey):
        """Get maximum x displacement for agent and prey to still intersect."""
        agent_x_vertices = agent.vertices[:, 0]
        agent_width = np.max(agent_x_vertices) - np.min(agent_x_vertices)
        prey_x_vertices = prey.vertices[:, 0]
        prey_width = np.max(prey_x_vertices) - np.min(prey_x_vertices)
        x_distance_threshold = 0.5 * (agent_width + prey_width)
        return x_distance_threshold

    def reward(self, state, meta_state, step_count):
        del step_count

        if meta_state['phase'] == self._phase and not self._reward_given and len(state['agent']) > 0:

            agent = state['agent'][0]
            prey = state['prey'][0]
            x_distance_threshold = self._get_x_distance_threshold(agent, prey)
            prey_exit_x = meta_state['prey_path'][-1][0]

            if np.abs(agent.x - prey_exit_x) < x_distance_threshold:
                self._reward_given = True
                return 1, False
            else:
                return 0, False
        else:
            return 0, False