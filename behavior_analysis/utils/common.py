"""Functions to show trial videos and images."""

import sys
sys.path.append('../../task')

import collections
import json
from matplotlib import path as mpl_path
from matplotlib import transforms as mpl_transforms
from moog import observers
from moog import sprite as sprite_lib
import numpy as np
import os

ATTRIBUTES_FULL = list(sprite_lib.Sprite.FACTOR_NAMES)

ATTRIBUTES_PARTIAL = ['x', 'y', 'opacity', 'metadata']

ATTRIBUTES_PARTIAL_INDICES = {k: i for i, k in enumerate(ATTRIBUTES_PARTIAL)}


def create_new_sprite(sprite_kwargs, vertices=None):
    """Create new sprite from factors.

    Args:
        sprite_kwargs: Dict. Keyword arguments for sprite_lib.Sprite.__init__().
            All of the strings in sprite_lib.Sprite.FACTOR_NAMES must be keys of
            sprite_kwargs.
        vertices: Optional numpy array of vertices. If provided, are used to
            define the shape of the sprite. Otherwise, sprite_kwargs['shape'] is
            used.

    Returns:
        Instance of sprite_lib.Sprite.
    """
    if vertices is not None:
        # Have vertices, so must invert the translation, rotation, and
        # scaling transformations to get the original sprite shape.
        center_translate = mpl_transforms.Affine2D().translate(
            -sprite_kwargs['x'], -sprite_kwargs['y'])
        x_y_scale = 1. / np.array([
            sprite_kwargs['scale'],
            sprite_kwargs['scale'] * sprite_kwargs['aspect_ratio']
        ])
        transform = (
            center_translate +
            mpl_transforms.Affine2D().rotate(-sprite_kwargs['angle']) +
            mpl_transforms.Affine2D().scale(*x_y_scale)
        )
        vertices = mpl_path.Path(vertices)
        vertices = transform.transform_path(vertices).vertices

        sprite_kwargs['shape'] = vertices

    return sprite_lib.Sprite(**sprite_kwargs)


def attributes_to_sprite(a):
    """Create sprite with given attributes."""
    attributes = {x: a[i] for i, x in enumerate(ATTRIBUTES_FULL)}
    if len(a) > len(ATTRIBUTES_FULL):
        vertices = np.array(a[-1])
    else:
        vertices = None
    return create_new_sprite(attributes, vertices=vertices)


def get_initial_state(trial):
    """Get initial state OrderedDict."""
    def _attributes_to_sprite_list(sprite_list):
        return [attributes_to_sprite(s) for s in sprite_list]
    
    state = collections.OrderedDict([
        (k, _attributes_to_sprite_list(v))
        for k, v in trial[0]
    ])
    
    return state


def update_state(state, step_string, rules=()):
    """Update the state in place given a step string."""
    # Deal with state changes
    meta_state = step_string[4][1]
    if meta_state['phase'] == 'fixation':
        state['fixation'][0].opacity = 255
    else:
        state['fixation'][0].opacity = 255
    if meta_state['phase'] not in ['iti', 'fixation']:
        state['screen'][0].opacity = 0
    
    # Loop through sprite layers
    for x in step_string[-1]:

        # Update sprite attributes
        if x[0] in ['response', 'eye', 'prey', ]:
            sprites = state[x[0]]
            for s, s_attr in zip(sprites, x[1]):
                attributes = {k: v for v, k in zip(s_attr, ATTRIBUTES_PARTIAL)}
                s.position = [attributes['x'], attributes['y']]
                s.opacity = attributes['opacity']

        # Vanish prey if necessary
        for rule in rules:
            rule.step(state, meta_state)


def observer():
    """Get a renderer."""
    observer = observers.PILRenderer(
        image_size=(256, 256),
        anti_aliasing=1,
        color_to_rgb='hsv_to_rgb',
    )
    return observer
    