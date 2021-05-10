"""Maze."""

import copy
from moog import sprite
import numpy as np

_DIRECTIONS_NAMED = {
    'N': (0, 1),
    'S': (0, -1),
    'E': (1, 0),
    'W': (-1, 0),
}
_DIRECTIONS = [np.array(x) for x in list(_DIRECTIONS_NAMED.values())]

class Maze():
    """Maze class."""

    def __init__(self, width, height, prey_path=()):
        """Constructor.
        
        Maze is a grid with walls between some of the cells.

        Indexing scheme: Width is component 0 of cell indices, height is
        component 1. Width is indexed left-to-right, height is indexed
        bottom-to-top.

        Args:
            width: Int. Width of the grid of maze cells.
            height: Int. Height of the grid of maze cells.
            prey_path: Iterable of int 2-tuples, which are indices of cells in
                the prey path.
        """
        self._width = width
        self._height = height
        self._construct_walls()

        self._construct_connected_components()
        self._set_prey_path(prey_path)

    def _construct_walls(self):
        """Construct self._walls_frozen and self._walls_temporary."""
        # Horizontal walls
        h_walls_temp = []
        h_walls_frozen = []
        for i in range(self._width):
            for j in range(self._height + 1):
                left_vertex = (i, j)
                right_vertex = (i + 1, j)
                if 0 < j < self._height:
                    h_walls_temp.append((left_vertex, right_vertex))
                else:
                    h_walls_frozen.append((left_vertex, right_vertex))

        # Vertical walls
        v_walls_temp = []
        v_walls_frozen = []
        for i in range(self._width + 1):
            for j in range(self._height):
                bottom_vertex = (i, j)
                top_vertex = (i, j + 1)
                if 0 < i < self._width:
                    v_walls_temp.append((bottom_vertex, top_vertex))
                else:
                    v_walls_frozen.append((bottom_vertex, top_vertex))

        self._walls_temporary = h_walls_temp + v_walls_temp
        self._walls_frozen = h_walls_frozen + v_walls_frozen

    def _construct_connected_components(self):
        """Put each cell in it's own connected component."""
        self._connected_components = {
            (i, j): i + j * self._width
            for i in range(self._width) for j in range(self._height)
        }

    def _wall_to_cells(self, wall):
        """Find wall's neighboring cells."""
        vertex_0, vertex_1 = wall
        if vertex_0[0] < vertex_1[0]:  # Horizontal wall
            cell_0 = (vertex_0[0], vertex_0[1] - 1)  # Bottom cell
            cell_1 = (vertex_0[0], vertex_0[1])  # Top cell
        else:  # Vertical wall
            cell_0 = (vertex_0[0] - 1, vertex_0[1])  # Left cell
            cell_1 = (vertex_0[0], vertex_0[1])  # Right cell

        return cell_0, cell_1

    def _cell_to_walls(self, cell):
        """Find cell's boundary walls."""
        i, j = cell
        corner_bottom_left = (i, j)
        corner_bottom_right = (i + 1, j)
        corner_top_left = (i, j + 1)
        corner_top_right = (i + 1, j + 1)

        walls = [
            (corner_bottom_left, corner_bottom_right),
            (corner_top_left, corner_top_right),
            (corner_bottom_left, corner_top_left),
            (corner_bottom_right, corner_top_right),
        ]

        return walls

    def _remove_wall(self, wall):
        """Remove a wall, returning whether succesful."""
        if wall in self._walls_frozen:
            raise ValueError(f'Wall {wall} is frozen so cannot be removed.')

        cell_0, cell_1 = self._wall_to_cells(wall)
        component_0 = self._connected_components[cell_0]
        component_1 = self._connected_components[cell_1]

        self._walls_temporary.remove(wall)
        if component_0 == component_1:
            # Creating a loop, so we reject
            self._walls_frozen.append(wall)
            return False
        else:
            # Removing the wall doesn't create loop, so remove it
            self._merge_components(component_0, component_1)
            return True

    def _merge_components(self, component_0, component_1):
        """Merge two connected components.
        
        Arbitrarily keep the index of component_0, overwriting the component_1
        cells.
        """
        for k, v in self._connected_components.items():
            if v == component_1:
                self._connected_components[k] = component_0

    def sample_distractors(self):
        """Sample the maze outside of the prey path.

        Iteratively removes walls until no more temporary walls are left to
        remove.
        """
        while len(self._walls_temporary) > 0:
            # Sample a temporary wall
            wall_index = np.random.randint(len(self._walls_temporary))
            wall_to_remove = self._walls_temporary[wall_index]
            self._remove_wall(wall_to_remove)

    def sample_distractor_exit(self,prey_path=()):
        """Sample distractor exit points around non-target side boundaries
        """

        walls_to_remove = []
        # identify non-target side
        tail = prey_path[-1]
        d = np.subtract(prey_path[-1],prey_path[-2]) #  prey_path[-1] - prey_path[-2]
        if tail[0] == 0 and tuple(d) == _DIRECTIONS_NAMED['W']:
            nontarget_side = ['E', 'N', 'S']
        elif (tail[0] == self._width - 1 and
              tuple(d) == _DIRECTIONS_NAMED['E']):
            nontarget_side = ['W', 'N', 'S']
        elif (tail[1] == self._height - 1 and
              tuple(d) == _DIRECTIONS_NAMED['N']):
            nontarget_side = ['W', 'E', 'S']
        elif tail[1] == 0 and tuple(d) == _DIRECTIONS_NAMED['S']:
            nontarget_side = ['W', 'E', 'N']

        # choose random wall except corners
        for direction in nontarget_side:
            if direction == 'E':
                vertex_x = self._width
                vertex_y = np.random.randint(1, self._height)  # not including corners
                wall_to_remove = ((vertex_x, vertex_y),(vertex_x, vertex_y+1))
            elif direction == 'W':
                vertex_x = 0
                vertex_y = np.random.randint(1, self._height)  # not including corners
                wall_to_remove = ((vertex_x, vertex_y), (vertex_x, vertex_y + 1))
            elif direction == 'N':
                vertex_y = self._height
                vertex_x = np.random.randint(1,self._width)  # not including corners
                wall_to_remove = ((vertex_x, vertex_y),(vertex_x+1, vertex_y))
            elif direction == 'S':
                vertex_y = 0
                vertex_x = np.random.randint(1,self._width)  # not including corners
                wall_to_remove = ((vertex_x, vertex_y),(vertex_x+1, vertex_y))

            # except entry points
            if np.array_equal(wall_to_remove,self._entry_wall):
                self.sample_distractor_exit(prey_path=prey_path)

            walls_to_remove.append(wall_to_remove)

        # remove chosen walls
        for wall in walls_to_remove:
            if wall in self._walls_frozen:
                self._walls_frozen.remove(wall)

    def _set_prey_path(self, prey_path):
        """Set the prey path.
        
        Removes walls in between cells in the prey path, and adds walls on the
        boundary of the prey path to self._walls_frozen.

        Removes maze periphery walls touching the first and last prey_path
        cells.

        Args:
            Prey_path: Iterable of int 2-tuples, indexes of cells comprising the
                prey path. Should be ordered, so the first and last elements are
                on the maze periphery.
        """

        walls_to_remove = []
        walls_to_freeze = []
        
        # First, add all cell boundary walls to walls_to_freeze
        for cell in prey_path:
            walls_to_freeze.extend(self._cell_to_walls(cell))
        walls_to_freeze = list(set(walls_to_freeze))

        # Now, remove border walls between consecutive cells
        for cell_0, cell_1 in zip(prey_path[:-1], prey_path[1:]):
            if cell_0[0] == cell_1[0]:  # Cells are vertically adjacent
                vertex_left = (cell_0[0], max(cell_0[1], cell_1[1]))
                vertex_right = (vertex_left[0] + 1, vertex_left[1])
                wall = (vertex_left, vertex_right)
            else:  # Cells are horizontally adjacent
                vertex_bottom = (max(cell_0[0], cell_1[0]), cell_0[1])
                vertex_top = (max(cell_0[0], cell_1[0]), cell_0[1] + 1)
                wall = (vertex_bottom, vertex_top)
            
            walls_to_freeze.remove(wall)
            walls_to_remove.append(wall)

        # Finally, remove the maze periphery walls touching the first and last
        # cell
        for cell in [prey_path[0], prey_path[-1]]:
            for wall in self._cell_to_walls(cell):
                if wall in self._walls_frozen:
                    walls_to_remove.append(wall)
                    walls_to_freeze.remove(wall)
                if np.array_equal(cell, prey_path[0]):
                    self._entry_wall = wall
                else:
                    self._target_wall = wall


        for wall in walls_to_remove + walls_to_freeze:
            if wall in self._walls_temporary:
                self._walls_temporary.remove(wall)
            if wall in self._walls_frozen:
                self._walls_frozen.remove(wall)
        
        self._walls_frozen.extend(walls_to_freeze)

    def to_sprites(self, wall_width, border_width, **sprite_factors):
        """Convert maze to list of sprites.

        Args:
            wall_width: Scalar. How wide the wall sprites should be.
            sprite_factors: Dict. Other attributes of the sprites (e.g. color,
                opacity, ...).
        """
        wall_width_box = np.array([
            [-0.5 * wall_width, -0.5 * wall_width],
            [-0.5 * wall_width, 0.5 * wall_width],
            [0.5 * wall_width, 0.5 * wall_width],
            [0.5 * wall_width, -0.5 * wall_width],
        ])
        sprites = []
        for (v_start, v_end) in self._walls_temporary + self._walls_frozen:
            start_box = wall_width_box + np.array([v_start])
            end_box = wall_width_box + np.array([v_end])
            bounds = np.concatenate((start_box, end_box), axis=0)
            x_min, y_min = np.min(bounds, axis=0)
            x_max, y_max = np.max(bounds, axis=0)
            sprite_shape = np.array([
                [x_min, y_min],
                [x_min, y_max],
                [x_max, y_max],
                [x_max, y_min],
            ])
            maze_span = max(self._width, self._height)
            sprite_shape /= maze_span
            sprite_shape = 0.5 + (1 - 2 * border_width) * (sprite_shape - 0.5)
            new_sprite = sprite.Sprite(
                x=0., y=0., shape=sprite_shape, **sprite_factors)
            sprites.append(new_sprite)
        
        return sprites