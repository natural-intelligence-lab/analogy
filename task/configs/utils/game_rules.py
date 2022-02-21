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

class CreateFakePrey(game_rules.AbstractRule):
    """Create agent."""

    def __init__(self, trial_init):
        self._trial_init = trial_init

    def step(self, state, meta_state):
        del meta_state
        self._trial_init.create_fake_prey(state)

class CreatePathPrey(game_rules.AbstractRule):
    """Create agent."""

    def __init__(self, trial_init):
        self._trial_init = trial_init

    def step(self, state, meta_state):
        del meta_state
        self._trial_init.create_path_prey(state)

class GlueAgent(game_rules.AbstractRule):

    def step(self, state, meta_state):
        del meta_state
        agent = state['agent'][0]
        agent.mass = np.inf

class GluePathPrey(game_rules.AbstractRule):

    def step(self, state, meta_state):
        del meta_state
        path_prey = state['path_prey'][0]
        path_prey.mass = np.inf

class DimPrey(game_rules.AbstractRule):
    """Modify sprites in a layer or set of layers.

    A filter can be applied to modify only sprites within the layer of interest
    that satisfy some condition.
    """

    def __init__(self, layers, modifier, sample_one=False, filter_fn=None):
        """Constructor.

        Args:
            layers: String or iterable of strings. Must be a key (or keys) in
                the environment state. Layer(s) in which sprites are modified.
            modifier: Function taking in a sprite and modifying it in place.
            sample_one: Bool. Whether to sample one sprite to modify if multiple
                satisfy filter_fn at a given step.
            filter_fn: Optional filter function. If specified must take in a
                sprite and return a bool saying whether to consider modifying
                that sprite.
        """
        if isinstance(layers, str):
            layers = [layers]
        self._layers = layers
        self._modifier = modifier
        self._sample_one = sample_one
        self._filter_fn = filter_fn

    def step(self, state, meta_state):
        """Apply rule to state."""
        # del meta_state
        
        sprites_to_modify = [s for k in (self._layers or []) for s in (state[k] or [])]
        # sprites_to_modify=[]
        # if self._layers is not None:
        #     for k in self._layers:
        #         if state is not None:
        #             for s in state[k]:
        #                     sprites_to_modify.append(s)
                

        if self._filter_fn:
            sprites_to_modify = list(
                filter(self._filter_fn, sprites_to_modify))

        if not sprites_to_modify:
            return

        if self._sample_one:
            sprites_to_modify = [np.random.choice(sprites_to_modify)]

        for sprite in sprites_to_modify:
            self._modifier(sprite,meta_state)
