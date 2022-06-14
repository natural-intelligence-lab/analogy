"""Maze."""
import pdb
import copy
from moog import sprite
import numpy as np
from matplotlib import pyplot as plt
import time
from matplotlib import patches as patches

_DIRECTIONS_NAMED = {
    'N': (0, 1),
    'S': (0, -1),
    'E': (1, 0),
    'W': (-1, 0),
}
_DIRECTIONS = [np.array(x) for x in list(_DIRECTIONS_NAMED.values())]

class Maze():
    """Maze class."""

    def __init__(self, width, height, prey_path=(), all_walls=None, prey_path_only=None, p_distract=1):
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
            all_walls: if true, entire maze has been specified and just pass walls to frozen
            prey_path_only: if true, not _construct_walls & _construct_connected_components and
                walls_temporary & wall_frazen are empty
            p_distract: PathPartialDistract
        """
        self._width = width
        self._height = height

        if all_walls is None: # for pathNoDistract
            # Must randomly generate the maze
            if prey_path_only is None:
                self._construct_walls(p_distract)
                self._construct_connected_components()
                self._set_prey_path(prey_path)
            else:
                self._walls_temporary =[]
                self._walls_frozen=[]
                self._connected_components=[]
                self._set_prey_path(prey_path)
        else:
            # Entire maze has been specified
            self._walls_frozen = all_walls
            self._walls_temporary = []

    def _construct_walls(self,p_distract):
        """Construct self._walls_frozen and self._walls_temporary."""
        # Horizontal walls
        h_walls_temp = []
        h_walls_frozen = []
        for i in range(self._width):
            for j in range(self._height + 1):
                left_vertex = (i, j)
                right_vertex = (i + 1, j)
                if 0 < j < self._height:
                    if np.random.rand() < p_distract:
                        h_walls_temp.append((left_vertex, right_vertex))
                else: # maze boundary
                    if np.random.rand() < p_distract:
                        h_walls_frozen.append((left_vertex, right_vertex))

        # Vertical walls
        v_walls_temp = []
        v_walls_frozen = []
        for i in range(self._width + 1):
            for j in range(self._height):
                bottom_vertex = (i, j)
                top_vertex = (i, j + 1)
                if 0 < i < self._width:
                    if np.random.rand() < p_distract:
                        v_walls_temp.append((bottom_vertex, top_vertex))
                else: # maze boundary
                    if np.random.rand() < p_distract:
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
        """Sample distractor exit points at South side
            remove every other grid including correct exit point
        """

        walls_to_remove = []
        # identify correct exit point
        tail = prey_path[-1]

        # remove from correct exit point to its right
        for grid in range(tail[0],self._width,2):
            vertex_y = 0
            vertex_x = grid
            wall_to_remove = ((vertex_x, vertex_y), (vertex_x + 1, vertex_y))
            walls_to_remove.append(wall_to_remove)

        # remove from correct exit point to its left
        for grid in range(np.mod(tail[0],2),tail[0],2):
            vertex_y = 0
            vertex_x = grid
            wall_to_remove = ((vertex_x, vertex_y), (vertex_x + 1, vertex_y))
            walls_to_remove.append(wall_to_remove)

        # remove chosen walls
        for wall in walls_to_remove:
            if wall in self._walls_frozen:
                self._walls_frozen.remove(wall)

    def sample_distractor_entry(self,prey_path=()):
        """Sample distractor exit points at North side
            remove every other grid including correct exit point
        """

        walls_to_remove = []
        # identify correct exit point
        tail = prey_path[0]

        # remove from correct exit point to its right
        for grid in range(tail[0],self._width,2):
            vertex_y = self._height
            vertex_x = grid
            wall_to_remove = ((vertex_x, vertex_y), (vertex_x + 1, vertex_y))
            walls_to_remove.append(wall_to_remove)

        # remove from correct exit point to its left
        for grid in range(np.mod(tail[0],2),tail[0],2):
            vertex_y = self._height
            vertex_x = grid
            wall_to_remove = ((vertex_x, vertex_y), (vertex_x + 1, vertex_y))
            walls_to_remove.append(wall_to_remove)

        # if entry is on left or right boundary, remove
        if tail[0] == 0 and tail[1] != (self._height - 1):
            for grid in range(np.mod(tail[1], 2), tail[1], 2):  # below entry
                vertex_x = 0  # left
                vertex_y = grid
                wall_to_remove = ((vertex_x, vertex_y), (vertex_x, vertex_y + 1))
                walls_to_remove.append(wall_to_remove)
            for grid in range(tail[1], self._height, 2):  # above entry
                vertex_x = 0  # left
                vertex_y = grid
                wall_to_remove = ((vertex_x, vertex_y), (vertex_x, vertex_y + 1))
                walls_to_remove.append(wall_to_remove)

        if tail[0] == (self._width-1) and tail[1]!=(self._height-1):
            for grid in range(np.mod(tail[1], 2), tail[1], 2):  # below entry
                vertex_x = self._width  # right
                vertex_y = grid
                wall_to_remove = ((vertex_x, vertex_y), (vertex_x , vertex_y + 1))
                walls_to_remove.append(wall_to_remove)
            for grid in range(tail[1], self._height, 2):  # above entry
                vertex_x = self._width
                vertex_y = grid
                wall_to_remove = ((vertex_x, vertex_y), (vertex_x , vertex_y + 1))
                walls_to_remove.append(wall_to_remove)

        # if it's prey path, keep it
        for cell in prey_path:
            for wall in self._cell_to_walls(cell):
                for wall2 in walls_to_remove:
                    if np.array_equal(wall,wall2):
                        walls_to_remove.remove(wall)

        # remove chosen walls
        for wall in walls_to_remove:
            if wall in self._walls_frozen:
                self._walls_frozen.remove(wall)

    def set_distractor_path(self, distractor_path):
        """Set the distractor path.

        Removes walls in between cells in the distractor path, and adds walls on the
        boundary of the distractor path to self._walls_frozen.

        Prey path is preserved by using _walls_frozen

        Args:
            distractor_path: Iterable of int 2-tuples, indexes of cells comprising the
                prey path. Should be ordered, so the first and last elements are
                on the maze periphery.
        """
        # Now, remove border walls between consecutive cells
        for cell_0, cell_1 in zip(distractor_path[:-1], distractor_path[1:]):
            if cell_0[0] == cell_1[0]:  # Cells are vertically adjacent
                vertex_left = (cell_0[0], max(cell_0[1], cell_1[1]))
                vertex_right = (vertex_left[0] + 1, vertex_left[1])
                wall = (vertex_left, vertex_right)
            else:  # Cells are horizontally adjacent
                vertex_bottom = (max(cell_0[0], cell_1[0]), cell_0[1])
                vertex_top = (max(cell_0[0], cell_1[0]), cell_0[1] + 1)
                wall = (vertex_bottom, vertex_top)

            if (wall not in self._walls_frozen) and (wall in self._walls_temporary):
                self._remove_wall(wall)
    
    def add_distractor_path(self, distractor_path,prey_path):
        """
        TBD: debug (stimuli contains white boxes)

        add the distractor path.

        adds walls on the boundary of the distractor path to self._walls_frozen
        if on prey path, no addition

        Args:
            distractor_path: Iterable of int 2-tuples, indexes of cells comprising the
                prey path. Should be ordered, so the first and last elements are
                on the maze periphery.
            prey_path: Iterable of int 2-tuples, indexes of cells comprising the
                prey path. Should be ordered, so the first and last elements are
                on the maze periphery.
            
        """
        walls_to_freeze = []
        
        # First, add all cell boundary walls to walls_to_freeze
        for cell in distractor_path:
            walls_to_freeze.extend(self._cell_to_walls(cell))
        walls_to_freeze = list(set(walls_to_freeze))

        # Now, remove border walls between consecutive cells
        for cell_0, cell_1 in zip(distractor_path[:-1], distractor_path[1:]):
            if cell_0[0] == cell_1[0]:  # Cells are vertically adjacent
                vertex_left = (cell_0[0], max(cell_0[1], cell_1[1]))
                vertex_right = (vertex_left[0] + 1, vertex_left[1])
                wall = (vertex_left, vertex_right)
            else:  # Cells are horizontally adjacent
                vertex_bottom = (max(cell_0[0], cell_1[0]), cell_0[1])
                vertex_top = (max(cell_0[0], cell_1[0]), cell_0[1] + 1)
                wall = (vertex_bottom, vertex_top)      

            walls_to_freeze.remove(wall)

        # Finally, remove the maze periphery walls touching the first and last cell
        for cell in [distractor_path[0], distractor_path[-1]]:
            for wall in self._cell_to_walls(cell):
                if wall[0][1]==wall[1][1]: # horizontal
                    if wall[0][1]==0 or wall[0][1]==self._height: # bottom or top
                        # pdb.set_trace()
                        if wall in walls_to_freeze:
                            walls_to_freeze.remove(wall)
                if wall[0][0] == wall[1][0]:  # vertical
                    if wall[0][0] == 0 or wall[1][0] == self._width:  # bottom or top
                        # pdb.set_trace()
                        if wall in walls_to_freeze:
                            walls_to_freeze.remove(wall)
                if wall[0]==0 or wall[0]==self._width: # left or right
                    if wall in walls_to_freeze:
                        walls_to_freeze.remove(wall)

        # get all cell boundary of prey path
        prey_walls=[]
        for cell in prey_path:
            prey_walls.extend(self._cell_to_walls(cell))
        prey_walls = list(set(prey_walls))

        # add to walls_frozen if not on prey path
        for wall in walls_to_freeze:
            if wall not in prey_walls:
                self._walls_frozen.append(wall)
                
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
                if wall[0][1]==wall[1][1]: # horizontal
                    if wall[0][1]==0 or wall[0][1]==self._height: # bottom or top
                        # pdb.set_trace()
                        if wall in walls_to_freeze:
                            walls_to_freeze.remove(wall)
                if wall[0][0] == wall[1][0]:  # vertical
                    if wall[0][0] == 0 or wall[1][0] == self._width:  # bottom or top
                        # pdb.set_trace()
                        if wall in walls_to_freeze:
                            walls_to_freeze.remove(wall)
                if wall[0]==0 or wall[0]==self._width: # left or right
                    if wall in walls_to_freeze:
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

    def to_circle(self,edge1,edge2,n_point):
        """generate circular path from edge1 [x,y] to edge 2 [x,y] in n_point
        """
        radius=abs(edge1[0]-edge2[0])



    def to_sprites(self,
                   wall_width,
                   cell_size,
                   bottom_border,
                   maze_walls_near_turn=None,
                   **sprite_factors):
        """Convert maze to list of sprites.

        Args:
            wall_width: Scalar. How wide the wall sprites should be.
            cell_size: Scalar. Width and height of maze cells.
            bottom_border: Scalar. How near to the bottom of the frame is the
                bottom of the maze.
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
        all_walls = self._walls_temporary + self._walls_frozen

        fig1, ax = plt.subplots(figsize=(5, 5))

        for i,(v_start, v_end) in enumerate(all_walls):
            start_box = wall_width_box + np.array([v_start])
            end_box = wall_width_box + np.array([v_end])
            bounds = np.concatenate((start_box, end_box), axis=0)
            x_min, y_min = np.min(bounds, axis=0)
            x_max, y_max = np.max(bounds, axis=0)

            if maze_walls_near_turn is not None:
                done=False
                for k,near_turn_wall in enumerate(maze_walls_near_turn):
                    id_near_turn = (v_start, v_end) in near_turn_wall
                    if not done:
                        if id_near_turn:
                            tmp_index = near_turn_wall.index((v_start, v_end))

                            if k==0: # ur_u > ul with 270 deg rotation
                                sprite_shape = np.array([
                                    [x_min, y_max], # top left
                                    [x_min, y_min], # bottom left
                                    [x_max, y_max+(y_max-y_min)], # bottom right
                                    [(x_min+x_max)/2, y_max+(y_max-y_min)], # top right
                                ])
                                done = True
                            elif k==1: # ur_r,
                                sprite_shape = np.array([])
                                done = True
                            elif k==2: # ul_u > DL
                                sprite_shape = np.array([
                                    [x_min, y_max],
                                    [x_min, y_min],
                                    [(x_min + x_max) / 2, y_min - (y_max - y_min)],
                                    [x_max, y_min - (y_max - y_min)],
                                ])
                                done = True
                            elif k==3: # ul_l,
                                sprite_shape = np.array([])
                                done = True
                            elif k==4: # dr_r > UR
                                sprite_shape = np.array([
                                    [x_min, y_max],
                                    [x_max+(x_max-x_min), y_min],
                                    [x_max+(x_max-x_min), (y_max + y_min)/2],
                                    [x_max, y_max],
                                ])
                                done = True
                            elif k==5: # dr_d,
                                sprite_shape = np.array([])
                                done = True
                            elif k==6: # dl_l > DR
                                sprite_shape = np.array([
                                    [x_min, y_min],
                                    [x_max, y_min],
                                    [x_max + (x_max - x_min), (y_max + y_min) / 2],
                                    [x_max + (x_max - x_min), y_max],
                                ])
                                done = True
                            elif k==7: # dl_d
                                sprite_shape =np.array([])
                                done = True
                        else: # id_near_turn
                            sprite_shape = np.array([
                                [x_min, y_min],
                                [x_min, y_max],
                                [x_max, y_max],
                                [x_max, y_min],
                            ])

            else:
                sprite_shape = np.array([
                    [x_min, y_min],
                    [x_min, y_max],
                    [x_max, y_max],
                    [x_max, y_min],
                ])

            # Shrink sprite to cell_size
            sprite_shape *= cell_size
            # Move sprite right to center maze on the x axis and up by
            # bottom_border
            total_width = cell_size * self._width
            if sprite_shape.shape[0]!=0:
                sprite_shape += np.array([[0.5 * (1 - total_width), bottom_border]])

                # plt.figure(1)
                ax.plot(np.append(sprite_shape[:,0],sprite_shape[0,0]), np.append(sprite_shape[:,1],sprite_shape[0,1]))
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                fig1.show()
                # fig1.canvas.draw()
                # fig1.canvas.flush_events()
                # time.sleep(0.1)
                # plt.show()
                # # patches.Polygon(sprite_shape)

                new_sprite = sprite.Sprite(
                    x=0., y=0., shape=sprite_shape, **sprite_factors)
                sprites.append(new_sprite)

        return sprites

    @property
    def walls(self):
        return self._walls_frozen + self._walls_temporary