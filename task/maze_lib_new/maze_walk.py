"""Maze walk physics."""

import numpy as np
from moog import physics as physics_lib
from maze_lib.maze import Maze

_EPSILON = 1e-4

_STOP_POINT = np.array([0.5, 0.1])
_STOP_PROXIMITY = 0.07


class MazeWalk(physics_lib.AbstractPhysics):
    """Maze physics class."""

    def __init__(self,
                 speed,
                 avatar_layer='prey',
                 start_lead_in=0.07):
        """Constructor.
        
        Args:
            speed: Float. Speed of avatar.
            avatar_layer: String. Layer of the avatar.
            start_lead_in: Float. How much lead-in to give for the start.
        """
        super(MazeWalk, self).__init__(updates_per_env_step=1)
        self.speed = speed
        self._avatar_layer = avatar_layer
        self._start_lead_in = start_lead_in

    def set_maze(self, maze_arms, prey_arm):
        maze = Maze(maze_arms)
        # vertices are the endpoints of each segment of the prey's arm
        self._vertices = [x[0] for x in maze.arms[prey_arm].segments[::-1]]
        self._vertices.append(maze.arms[prey_arm].segments[0][1])
        self._direction_arrays = [
            -1 * x for x in maze.arms[prey_arm].direction_arrays[::-1]]
        self._current_segment = 0
        self._reset_sprite_position = True

    def _update_sprite_in_maze(self, sprite):
        """Update a sprite velocity to abide by the maze."""

        pos = sprite.position
        d = self._direction_arrays[self._current_segment]
        sprite.angle = -np.pi/2*d[0]  # for reward; d[1]=1,-1 for (r,l); 12 o'clock = degree zero & CCW for +

        # Reset the sprite's position if first step
        if self._reset_sprite_position:
            self._reset_sprite_position = False
            start_vertex = self._vertices[0]
            sprite.position = start_vertex - self._start_lead_in * d
            return

        if np.isinf(sprite.mass):
            return
        
        if self.speed == 0.:
            return

        # Increment sprite position
        vel = d * self.speed
        new_pos = pos + vel

        # If on the last segment, keep drifting
        if self._current_segment == len(self._vertices) - 2:
            sprite.position = new_pos
            return

        # Check if should enter next segment
        end_vertex = self._vertices[self._current_segment + 1]
        vertex_diff = new_pos - end_vertex
        dot = np.dot(vertex_diff, d)
        if dot > 0:
            overshoot = np.sum(np.abs(vertex_diff))
            next_d = self._direction_arrays[self._current_segment + 1]
            new_pos = end_vertex + overshoot * next_d
            self._current_segment += 1

        sprite.position = new_pos

    def apply_physics(self, state, updates_per_env_step):
        """Move the sprites according to the physics."""
        del updates_per_env_step
        for sprite in state[self._avatar_layer]:
            self._update_sprite_in_maze(sprite)

    @property
    def start_lead_in(self):
        return self._start_lead_in