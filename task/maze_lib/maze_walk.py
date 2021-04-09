"""Maze walk physics."""

import numpy as np
from moog import physics as physics_lib

_EPSILON = 1e-4


class MazeWalk(physics_lib.AbstractPhysics):
    """Maze physics class."""

    def __init__(self,
                 speed,
                 avatar_layer='prey',
                 start_lead_in=0.07,
                 stop_center_proximity=0.07):
        """Constructor.
        
        Args:
            speed: Float. Speed of avatar.
            avatar_layer: String. Layer of the avatar.
            start_lead_in: Float. How much lead-in to give for the start.
            stop_center_proximity: Float. How close to (0.5, 0.5) to stop.
        """
        super(MazeWalk, self).__init__(updates_per_env_step=1)
        self._speed = speed
        self._avatar_layer = avatar_layer
        self._start_lead_in = start_lead_in
        self._stop_center_proximity = stop_center_proximity

    def set_maze(self, maze_arms, avatar_arm):
        arm = maze_arms[avatar_arm]
        vertex = 0.5 * np.ones(2)
        vertices = [vertex]
        directions = []
        for (d, l) in arm:
            vertex = vertex + d * l
            vertices.append(vertex)
            # Negative because avatar moves from end to start on arm
            directions.append(-1 * d)

        # Reverse vertices and directions because avatar moves from end to start
        # on arm
        self._vertices = vertices[::-1]
        self._directions = directions[::-1]
        self._current_segment = 0
        self._reset_sprite_position = True

    def _update_sprite_in_maze(self, sprite):
        """Update a sprite velocity to abide by the maze."""

        pos = sprite.position
        d = self._directions[self._current_segment]
        
        # Reset the sprite's position if first step
        if self._reset_sprite_position:
            self._reset_sprite_position = False
            start_vertex = self._vertices[0]
            sprite.position = start_vertex - self._start_lead_in * d
            return

        if np.isinf(sprite.mass):
            return
        
        if self._speed == 0.:
            return

        # Check if reached center of screen and should stop
        if np.linalg.norm(pos - 0.5) < self._stop_center_proximity:
            return

        # Increment sprite position
        vel = d * self._speed
        new_pos = pos + vel

        # Check if should enter next segment
        end_vertex = self._vertices[self._current_segment + 1]
        vertex_diff = new_pos - end_vertex
        dot = np.dot(vertex_diff, d)
        if dot > 0:
            overshoot = np.sum(np.abs(vertex_diff))
            next_d = self._directions[self._current_segment + 1]
            new_pos = end_vertex + overshoot * next_d
            self._current_segment += 1

        sprite.position = new_pos

    def apply_physics(self, state, updates_per_env_step):
        """Move the sprites according to the physics."""
        del updates_per_env_step
        for sprite in state[self._avatar_layer]:
            self._update_sprite_in_maze(sprite)