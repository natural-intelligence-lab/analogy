"""Chase/avoid task with predators, prey and an occluder.

The predators (red circles) chase the agent. The agent receives reward for
catching prey (yellow circles), which disappear upon capture. There is an
occluder (gray rectangle) in the arena.
"""

import collections
import numpy as np
import os

from moog import action_spaces
from moog import observers
from moog.observers import polygon_modifiers
from moog import physics as physics_lib
from moog import game_rules
from moog import sprite
from moog import tasks
from moog.state_initialization import distributions as distribs
from moog.state_initialization import sprite_generators


def _get_config():
    """Get environment config."""

    ############################################################################
    # Sprite initialization
    ############################################################################

    def state_initializer():
        
        agent_big = sprite.Sprite(
            x=0., y=0., c0=0., c1=1., c2=1., shape='circle', scale=0.02,
            opacity=0,
        )

        small_cross = 0.1 * np.array([
            [-5, 1], [-1, 1], [-1, 5], [1, 5], [1, 1], [5, 1], [5, -1], [1, -1],
            [1, -5], [-1, -5], [-1, -1], [-5, -1],
        ])
        agent_small = sprite.Sprite(
            x=0., y=0., c0=0., c1=0., c2=1., shape=small_cross, scale=0.01,
            opacity=0,
        )

        state = collections.OrderedDict([
            ('agent', [agent_big, agent_small]),
        ])

        return state

    ############################################################################
    # Physics
    ############################################################################

    physics = physics_lib.Physics(
        updates_per_env_step=1,
    )

    ############################################################################
    # Task
    ############################################################################

    task = tasks.Reset(condition=lambda state, meta_state: False)

    ############################################################################
    # Action space
    ############################################################################

    action_space = action_spaces.SetPosition(
        action_layers=('agent',),
        inertia=0.,
    )

    ############################################################################
    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(256, 256),
        anti_aliasing=1,
        color_to_rgb=observers.color_maps.hsv_to_rgb,
    )

    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initializer,
        'physics': physics,
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer, 'state': observers.RawState()},
        'game_rules': (),
    }
    return config


def get_config(_):
    return _get_config()