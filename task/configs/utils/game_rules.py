"""Custom game rules."""

from moog import game_rules


class CreateAgent(game_rules.AbstractRule):
    """Create agent."""

    def __init__(self, trial_init):
        self._trial_init = trial_init

    def step(self, state, meta_state):
        del meta_state
        self._trial_init.create_agent(state)
