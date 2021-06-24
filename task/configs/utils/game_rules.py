"""Custom game rules."""

from moog import game_rules
import numpy as np


class CreateAgent(game_rules.AbstractRule):
    """Create agent."""

    def __init__(self, trial_init):
        self._trial_init = trial_init

    def step(self, state, meta_state):
        del meta_state
        self._trial_init.create_agent(state)



class GlueAgent(game_rules.AbstractRule):
    """Create agent."""

    def step(self, state, meta_state):
        del meta_state
        agent = state['agent'][0]
        agent.mass = np.inf

