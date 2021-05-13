"""Custom tasks."""

from moog import tasks
import inspect
from dm_env import specs
import numpy as np


class PressWhenReady(tasks.AbstractTask):
    """Press When Ready task for offline.
        response key: space bar
        maximum_duration: response window
    """

    def __init__(self,condition,
                 maximum_duration=np.inf):
        """Constructor.

        Args:
            condition: Function with one of the following signatures:
                    * state --> bool
                    * state, meta_state --> bool
                The bool is whether to reset.
            maximum_duration: Int. Number of steps after condition is True
                to reset.
        """
        if len(inspect.signature(condition).parameters.values()) == 1:
            self._condition = lambda state, meta_state: condition(state)
        else:
            self._condition = condition

        self._maximum_duration = maximum_duration


    def reset(self, state, meta_state):
        # We reset to infinity, because self._steps_until_reset will be
        # decremented every time self.reward() is called, so is only set to a
        # finite value when the condition is met and reset is imminent.
        self._steps_until_reset = np.inf

        del state
        del meta_state

    def reward(self, state, meta_state, step_count):
        del step_count
        if (self._steps_until_reset == np.inf and
                self._condition(state, meta_state)):  # 1st after reset & condition
            reward = 0
            self._steps_until_reset = self._maximum_duration
        else:
            reward = 0.

        self._steps_until_reset -= 1
        should_reset = self._steps_until_reset < 0

        # if responded
        response = [s for s in state['responses_offline'] if s.opacity > 0]
        if (len(response) > 0 and
                self._condition(state, meta_state)):
            return 0, 0

        return reward, should_reset

