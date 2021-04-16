"""Maze."""

import copy
from maze_lib import constants
from moog import sprite
import numpy as np

_EPSILON = 1e-4
_MAX_ITERS = int(1e4)
_DEFAULT_END = (0.5, 0.1)


def _opposite_direction(d):
    if d == 'u':
        return 'd'
    elif d == 'd':
        return 'u'
    elif d == 'l':
        return 'r'
    elif d == 'r':
        return 'l'
    else:
        raise ValueError(f'Direction {d} not recognized')


class MazeArm():
    """Maze arm object."""

    def __init__(self, directions, lengths, end=_DEFAULT_END):
        """
        TODO(nwatters): Add documentation.
        """
        # Convert directions to list
        directions = list(directions)

        # Sanity check to make sure directions and lengths have same number
        if len(directions) != len(lengths):
            raise ValueError(
                f'directions {directions} must have the same length as lengths '
                f'{lengths}'
            )

        # Sanity check to make sure directions never reverse
        for d_0, d_1 in zip(directions[:-1], directions[1:]):
            if d_1 == d_0 or d_1 == _opposite_direction(d_0):
                raise ValueError(
                    f'Have invalid consecutive directions {d_0} and {d_1} in '
                    f'directions {directions}'
                )

        self._end = end

        self.lengths = lengths
        self.direction_strings = directions
        self.direction_arrays = [
            getattr(constants.Directions, d) for d in directions]

        self._segments = self._get_segments(self.direction_arrays, lengths, end)

    def _get_segments(self, direction_arrays, lengths, end):
        """TODO(nwatters): Add documentation."""
        segments = []
        v_end = np.array(end)
        for (d, l) in zip(direction_arrays, lengths):
            v_start = v_end + d * l
            new_segment = (v_start, v_end)
            segments.append(new_segment)
            v_end = copy.copy(v_start)
        
        return segments

    def to_sprites(self, arm_width, **sprite_factors):
        """Convert arm to list of sprites.

        Args:
            arm_width: Scalar. How wide the arm segments should be.
            sprite_factors: Dict. Other attributes of the sprites (e.g. color,
                opacity, ...).
        """
        arm_width_box = np.array([
            [-0.5 * arm_width, -0.5 * arm_width],
            [-0.5 * arm_width, 0.5 * arm_width],
            [0.5 * arm_width, 0.5 * arm_width],
            [0.5 * arm_width, -0.5 * arm_width],
        ])
        sprites = []
        for (v_start, v_end) in self._segments:
            start_box = arm_width_box + np.array([v_start])
            end_box = arm_width_box + np.array([v_end])
            bounds = np.concatenate((start_box, end_box), axis=0)
            x_min, y_min = np.min(bounds, axis=0)
            x_max, y_max = np.max(bounds, axis=0)
            sprite_shape = np.array([
                [x_min, y_min],
                [x_min, y_max],
                [x_max, y_max],
                [x_max, y_min],
            ])
            new_sprite = sprite.Sprite(
                x=0., y=0., shape=sprite_shape, **sprite_factors)
            sprites.append(new_sprite)
        
        return sprites

    @property
    def segments(self):
        return self._segments


class Maze():

    def __init__(self, arms=()):
        """Constructor.

        Args:
            arms: Iterable of arms. Each arm is a dictionary which may have keys
                ['end', 'directions', 'lengths']. Here 'end' is a (x, y)
                pair for the end of the arm, 'directions' is a string with
                ['u', 'd', 'l', 'r'] characters specifying the direction of each
                segment, and 'lengths' is an iterable of scalars specifying the
                length of each segment. The segments are listed end-to-start.
        """
        self._arms = [MazeArm(**x) for x in arms]

    def to_sprites(self, arm_width, **sprite_factors):
        sprites = []
        for arm in self._arms:
            sprites.extend(arm.to_sprites(arm_width, **sprite_factors))
        return sprites

    @property
    def arms(self):
        return self._arms
