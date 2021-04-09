"""Maze."""

from maze_lib import constants
from moog import sprite
import numpy as np

_EPSILON = 1e-4
_MAX_ITERS = int(1e4)


class Maze():

    def __init__(self, arms=(), separation_threshold=0):
        """Constructor.

        Args:
            arms: Iterable of arms. Each arm is an iterable of
                (direction, length) pairs.
            separation_threshold: Scalar. Threshold for minimum distance between
                arms.
        """
        self._separation_threshold = separation_threshold

        self._arms = []
        self._arm_segments = []
        self._segments = []

        for arm in arms:
            valid_arm = self.add_arm(arm)
            if not valid_arm:
                raise ValueError(
                    f'Arm {arm} does not respect separation_threshold '
                    f'{separation_threshold}')

    def add_arm(self, arm):
        """Add an arm."""
        segments = []
        vertex = np.zeros(2)
        for (d, l) in arm:
            new_v = vertex + d * l
            if not all(self._valid_vertex(s, new_v) for s in self._segments):
                return False
            new_segment = (vertex, new_v)
            if not all(self._valid_vertex(new_segment, s[1])
                       for s in self._segments):
                return False

            vertex = new_v
            segments.append(new_segment)
        
        self._arms.append(arm)
        self._arm_segments.append(segments)
        self._segments.extend(segments)
        return True

    def _valid_vertex(self, segment, vertex):
        """Check if a vertex is valid with respect to a segment."""
        start, end = segment
        end_tilde = end - start
        norm = np.linalg.norm(end_tilde)
        v_tilde = (vertex - start) / norm
        dot = np.dot(v_tilde, end_tilde) / norm
        dot = np.clip(dot, 0, 1)
        nearest_point = norm * v_tilde - dot * end_tilde
        nearest_dist = np.linalg.norm(nearest_point)
        if nearest_dist < self._separation_threshold:
            return False
        else:
            return True

    def _sample_arm(self, direction, num_segments, allow_continuation=True):
        """Sample random arm."""
        arm = []
        for i in range(num_segments):
            if i > 0:
                if allow_continuation:
                    available_directions = [
                        d for d in constants.Directions
                        if np.dot(d, direction) > -1 * _EPSILON
                    ]
                else:
                    available_directions = [
                        d for d in constants.Directions
                        if _EPSILON > np.dot(d, direction) > -1 * _EPSILON
                    ]
                direction = available_directions[
                    np.random.randint(len(available_directions))]
            
            length = constants.ArmLengths.sample()
            arm.append((direction, length))
        
        return arm

    def sample_arm(self, direction, num_segments, allow_continuation=True):
        for _ in range(_MAX_ITERS):
            arm = self._sample_arm(
                direction, num_segments, allow_continuation=allow_continuation)
            if self.add_arm(arm):
                return arm
        return False

    def to_sprites(self, arm_width, **sprite_factors):
        arm_width_box = np.array([
            [-1 * arm_width, -1 * arm_width],
            [-1 * arm_width, arm_width],
            [arm_width, arm_width],
            [arm_width, -1 * arm_width],
        ])
        sprites = []
        for arm_segments in self._arm_segments:
            for start, end in arm_segments:
                start_box = arm_width_box + np.array([start])
                end_box = arm_width_box + np.array([end])
                bounds = np.concatenate((start_box, end_box), axis=0)
                x_min, y_min = np.min(bounds, axis=0)
                x_max, y_max = np.max(bounds, axis=0)
                sprite_shape = 0.5 + np.array([
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
    def arms(self):
        return self._arms

    @property
    def arm_segments(self):
        return self._arm_segments
