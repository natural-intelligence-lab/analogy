"""Conditions."""

import numpy as np


class VerticalTiming():

    def __init__(self):
        self._maze_width = 11  # Reasonable to make the maze a good size
        self._min_height = 2
        self._max_height = self._maze_width
        self._heights = range(self._min_height, self._max_height + 1)

    def _sample_condition(self, height, x):
        # # exclude trials where x is near initial paddle position (0.5)
        # if np.abs(x-self._maze_width/2) < 1:
        #     return []

        prey_path = [[x, i] for i in range(height, -1, -1)]

        maze_walls_left = [((x, i), (x, i + 1)) for i in range(height)]
        maze_walls_right = [((x + 1, i), (x + 1, i + 1)) for i in range(height)]
        maze_walls = maze_walls_left + maze_walls_right

        features = {
            'name': 'VerticalTiming',
            'path_length': height,
            'x': x,
        }
        maze_width = self._maze_width
        maze_height = height
        condition = [maze_width, maze_height, prey_path, maze_walls, features]
        return condition

    def __call__(self):
        conditions = [
            self._sample_condition(height, x)
            for height in self._heights
            for x in range(self._maze_width)]
        return conditions
