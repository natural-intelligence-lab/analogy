"""Classes to load and sample stimuli."""

import copy
import json
from layered_maze_lib import maze_composer
from layered_maze_lib import path_dataset
from wire_maze_lib import maze_composer as wire_maze_composer
from wire_maze_lib import path_dataset as wire_path_dataset
import logging
import numpy as np
import os
from scipy import signal as scipy_signal
from matplotlib import pyplot as plt

class Sampler():
    """Stimulus sampler class.

    This should serve as the `stimulus_generator` argument to a Config().
    """
    
    def __init__(self,
                 stimuli_dir,
                 filter_fn=None,
                 length=np.inf,
                 num_passes=1):
        """Constructor.
        
        Args:
            TODO(nwatters): Add documentation.
        """
        self._length = length
        self._num_passes = num_passes

        if filter_fn is None:
            self._filter_fn = lambda _: True
        else:
            self._filter_fn = filter_fn

        self._load_stimuli(stimuli_dir)

        self._count = 0
        self._pass_num = -1
        self._reset_cycle()

    def _reset_cycle(self):
        """Looped through all stimuli, so re-sampler ordering and loop again."""
        self._cycle = np.random.permutation(len(self._stimuli))
        self._pass_num += 1

    def _process_stimulus_string(self, x):
        """Convert stimulus string from logs into a stimulus for the config."""
        maze_width, maze_height, prey_path, maze_walls, features = x
        if not self._filter_fn(features):
            return []

        stimulus = dict(
            maze_width=maze_width,
            maze_height=maze_height,
            prey_path=prey_path,
            maze_walls=maze_walls,
            features=features,
        )
        return [stimulus]

    def _load_stimuli(self, stimuli_dir):
    
        stimulus_filenames = sorted(
            filter(lambda s: s.isnumeric(), os.listdir(stimuli_dir)))
        stimulus_strings = [
            json.load(open(os.path.join(stimuli_dir, x)))
            for x in stimulus_filenames
        ]

        self._stimuli = []
        for x in stimulus_strings:
            self._stimuli.extend(self._process_stimulus_string(x))

        if len(self._stimuli) == 0:
            raise ValueError(
                'No stimuli. Check your filter_fn argument.')

        if self._length < len(self._stimuli):
            self._stimuli = self._stimuli[:self._length]

    def __call__(self):
        """Return stimulus for the config."""
        
        if self._pass_num == self._num_passes:
            # Ran out of stimuli
            return None

        ind = self._cycle[self._count]
        self._count += 1

        if self._count == len(self._stimuli):
            # Finished a cycle through all the stimuli, so begin another cycle
            self._reset_cycle()
            self._count = 0

        stimulus = copy.deepcopy(self._stimuli[ind])

        return stimulus

    def __len__(self):
        return self._num_passes * len(self._stimuli)


class MixtureSampler():

    def __init__(self, *samplers, num_passes=1):
        
        self._samplers = samplers
        self._num_passes = num_passes

        # Create indices for which sampler to use at each trial
        sampler_inds = []
        for i, x in enumerate(samplers):
            sampler_inds.extend(len(x) * [i])
        self._sampler_inds = [
            sampler_inds[i] for i in np.random.permutation(len(sampler_inds))]

        self._count = 0
        self._pass_num = -1

    def _reset_cycle(self):
        """Looped through all stimuli, so re-sampler ordering and loop again."""
        self._pass_num += 1
        # Create indices for which sampler to use at each trial
        sampler_inds = []
        for i, x in enumerate(self._samplers):
            sampler_inds.extend(len(x) * [i])
        self._sampler_inds = [
            sampler_inds[i] for i in np.random.permutation(len(sampler_inds))]

    def __call__(self):
        if self._pass_num == self._num_passes:
            # Ran out of stimuli
            return None
        
        if self._count >= len(self._sampler_inds):
            # Finished a cycle through all the stimuli, so begin another cycle
            self._reset_cycle()
            self._count = 0
            # return None

        sampler_ind = self._sampler_inds[self._count]
        self._count += 1
        return self._samplers[sampler_ind]()

    def __len__(self):
        return self._num_passes * len(self._sampler_inds)


class ChainedSampler():

    def __init__(self, *samplers):
        self._samplers = samplers
        self._sampler_ind = 0

    def __call__(self):
        sampler = self._samplers[self._sampler_ind]
        stimulus = sampler()
        if stimulus is None:
            self._sampler_ind += 1
            if self._sampler_ind >= len(self._samplers):
                return None
            else:
                return self()
        else:
            return stimulus

    def __len__(self):
        return sum([len(x) for x in self._samplers])


class LayeredMazeSampler(maze_composer.MazeComposer):
    """Generates stimuli composed of overlaying paths."""

    def __init__(self,
                 path_dir,
                 num_layers,
                 ball_path_top_bottom=False,
                 ball_path_top=True,
                 max_num_turns=np.inf,
                 num_turns=None):
        """Constructor.
        
        Args:
            path_dir: String. Directory of path dataset to use for composing
                mazes.
            num_layers: Int. Number of paths to compose for each maze.
                Equivalently, one greater than number of distractor paths.
            ball_path_top_bottom: Bool. Whether the ball path should be forced
                to enter from the top and exit from the bottom.
            ball_path_top: Bool. Whether the ball path should be forced
                to enter from the top.
            max_num_turns: Int. Maximum number of turns for the ball path.
            num_turns: Int. if not None, number of turns for the ball path.
        """
        super(LayeredMazeSampler, self).__init__(
            path_dir=path_dir,
            num_layers=num_layers,
            pixels_per_square=2,
            ball_path_top_bottom=ball_path_top_bottom,
            ball_path_top=ball_path_top,
            max_num_turns=max_num_turns,
            num_turns = num_turns,
        )

    def _get_maze_walls(self, maze):
        kernel_v = np.array([[1], [1], [1]])
        kernel_h = np.array([[1, 1, 1]])
        conv_v = scipy_signal.convolve2d(maze, kernel_v, mode='same', boundary='symm')
        conv_h = scipy_signal.convolve2d(maze, kernel_h, mode='same', boundary='symm')
        walls_v = conv_v[::2, 1::2] == 3.
        walls_h = conv_h[1::2, ::2] == 3.

        walls_v = [
            (tuple(x + np.array([0, 1])), tuple(x + np.array([1, 1])))
            for x in np.argwhere(walls_v)
        ]
        walls_h = [
            (tuple(x + np.array([1, 0])), tuple(x + np.array([1, 1])))
            for x in np.argwhere(walls_h)
        ]
        return walls_v + walls_h

    def _num_turns_path(self, prey_path):
         path_x = prey_path[:, 0]
         path_x_diff = path_x[1:] - path_x[:-1]
         num_turns = np.sum(
             np.abs(np.convolve(path_x_diff, [-1, 1], mode='valid')))  # detect change point

         return num_turns

    def __call__(self):
        maze, path = super(LayeredMazeSampler, self).__call__()

        maze, path = path_dataset.rotate_maze_and_path_90(
            maze, path, num_times=3)

        maze_width = int((maze.shape[0] + 1) / 2)
        maze_height = int((maze.shape[1] + 1) / 2)
        maze_walls = self._get_maze_walls(maze)
        prey_path = [tuple(x) for x in (path[::2] / 2).astype(int)]

        num_turns = self._num_turns_path(path)

        features = {
            'name': 'LayeredMaze',
            'start_x': prey_path[0][1],
            'num_turns': num_turns,
            'path_length': len(prey_path),
        }

        stimulus = dict(
            maze_width=maze_width,
            maze_height=maze_height,
            prey_path=prey_path,
            maze_walls=maze_walls,
            features=features,
        )

        return stimulus

    def __len__(self):
        return self._num_mazes  # np.inf


class WireMazeSampler(wire_maze_composer.MazeComposer):
    """Generates stimuli composed of overlaying paths."""

    def __init__(self,
                 path_dir,
                 num_layers,
                 ball_path_top_bottom=True,
                 distractors_top_bottom=True,
                 min_num_turns=0,
                 max_num_turns=np.inf,
                 min_num_overlap=0,
                 max_num_overlap=np.inf,
                 min_exit_distance=0,
                 distractors_num_turns=None):
        """Constructor.

        Args:
            path_dir: String. Directory of path dataset to use for composing
                mazes.
            num_layers: Int. Number of paths to compose for each maze.
                Equivalently, one greater than number of distractor paths.
            ball_path_top_bottom: Bool. Whether the ball path should be forced
                to enter from the top and exit from the bottom.
            distractors_top_bottom: Bool. Whether all distractor paths should be
                forced to enter from the top and exit from the bottom.
            min_num_turns: Int. Minimum number of turns for the ball path.
            max_num_turns: Int. Maximum number of turns for the ball path.
            min_num_overlap: Int. impose distractors have crossed the ball path with this number
            min_exit_distance: Int. impose contraint of exits between path and distriactors being large than min
        """
        super(WireMazeSampler, self).__init__(
            path_dir=path_dir,
            num_layers=num_layers,
            pixels_per_square=2,  # Irrelevant when using MOOG
            ball_path_top_bottom=ball_path_top_bottom,
            distractors_top_bottom=distractors_top_bottom,
            min_num_turns=min_num_turns,
            max_num_turns=max_num_turns,
            min_num_overlap=min_num_overlap,
            max_num_overlap=max_num_overlap,
            min_exit_distance=min_exit_distance,
            distractors_num_turns=distractors_num_turns,
        )

        self._gap_size = 0 # 0.15  # assuming wall size is one; if set to zero, no gap

    def _get_wall_touch_path(self, maze, path):
        wall_touch_path_0 = []  # short start
        wall_touch_path_1 = []  # short end
        start_path = path[:-1]
        end_path = path[1:]

        # from start/top
        for i, (start,end) in enumerate(zip(start_path,end_path)):
            if start[0] % 2 == 0 and end[0] % 2 ==0: # Horizontal wall; after rotation
                left_wall = [np.int((start[0] + end[0]) / 2), start[1] - 1] # left only in maze,  actually vertical(up)
                right_wall = [np.int((start[0] + end[0]) / 2), start[1] + 1]
                if maze[left_wall[0],left_wall[1]] > 0 and maze[right_wall[0],right_wall[1]] > 0: # path crossing
                    wall_touch_path_0.append(right_wall)
                    wall_touch_path_1.append(left_wall)
            if start[0] % 2 != 0 and end[0] % 2 != 0:  # Vertical wall
                up_wall = [start[0] - 1, np.int((start[1] + end[1]) / 2)]
                down_wall = [start[0] + 1, np.int((start[1] + end[1]) / 2)]
                if maze[up_wall[0],up_wall[1]] > 0 and maze[down_wall[0],down_wall[1]] > 0:  # path crossing
                    wall_touch_path_0.append(down_wall)
                    wall_touch_path_1.append(up_wall)
        return wall_touch_path_0,wall_touch_path_1

    def _add_gap_distractor_junction(self,walls):
        wall_array = np.asarray(walls)  # (# list,2,2)
        done=np.zeros((wall_array.shape[0],wall_array.shape[1]),dtype=bool)
        for i, wall in enumerate(walls):
            for j, edge in enumerate(wall):
                id_same_edge = np.logical_and(np.sum(np.abs(wall_array - edge), axis=2) == 0 ,(done==False)) # [# list, 2]
                n_same_edge = np.count_nonzero(id_same_edge)

                if n_same_edge > 2: # junction; note this is after inserting gap around target path, so no worry for them
                    done += id_same_edge

                    # build indices
                    i_edge = []
                    for i, row in enumerate(id_same_edge):
                        for j, val in enumerate(row):
                            if val:
                                i_edge.insert(0, [i, j])
                    wall_edge = wall_array[np.asarray(i_edge)[:, 0], :, :]  # (3/4,2,2)

                    id_horizontal = np.squeeze(np.diff(wall_edge, n=1, axis=1)[:, :, 0]>0).tolist()
                    id_start_junction = (np.asarray(i_edge)[:, 1] == 0).tolist()  # start is at junction; either RightHorizontal, TopVertical

                    # inserting gap
                    if np.random.random() > 0.5:  # vertical gap
                        for i_horizon,value in enumerate(id_horizontal):
                            x = i_edge[i_horizon][0]
                            if not value:
                                if id_start_junction[i_horizon]:  # TopVertical
                                    walls[x]=((walls[x][0][0],walls[x][0][1]+ self._gap_size), walls[x][1])
                                else: # bottomVertical
                                    walls[x] = (walls[x][0], (walls[x][1][0],walls[x][1][1] - self._gap_size))
                    else:  # horizontal gap
                        for i_horizon,value in enumerate(id_horizontal):
                            x = i_edge[i_horizon][0]
                            if value:
                                if id_start_junction[i_horizon]:  # RightHorizontal
                                    walls[x]=((walls[x][0][0]+ self._gap_size,walls[x][0][1]), walls[x][1])
                                else: # LeftHorizontal
                                    walls[x] = (walls[x][0], (walls[x][1][0] - self._gap_size, walls[x][1][1]))
        return walls


    def _get_maze_walls(self, maze, wall_touch_path_0, wall_touch_path_1):
        walls = []
        for i, row in enumerate(maze):
            for j, x in enumerate(row):
                if x > 0:  # == 1:  # Wall exists here
                    if i % 2 == 1:  # Vertical wall
                        if [i,j] in wall_touch_path_0:  # shorter start
                            start = (i / 2, (j - 1) / 2 + self._gap_size)
                            end = (i / 2, (j + 1) / 2)
                        elif [i, j] in wall_touch_path_1:  # shorter end
                            start = (i / 2, (j - 1) / 2)
                            end = (i / 2, (j + 1) / 2 - self._gap_size)
                        else:
                            start = (i / 2, (j - 1) / 2)
                            end = (i / 2, (j + 1) / 2)
                    else:  # Horizontal wall
                        if [i,j] in wall_touch_path_0:  # shorter start
                            start = ((i - 1) / 2 + self._gap_size, j / 2)
                            end = ((i + 1) / 2, j / 2)
                        elif [i, j] in wall_touch_path_1:  # shorter end
                            start = ((i - 1) / 2, j / 2)
                            end = ((i + 1) / 2 - self._gap_size, j / 2)
                        else:  # for distractors
                            start = ((i - 1) / 2, j / 2)
                            end = ((i + 1) / 2, j / 2)

                    # # Adjust for offset to make walls in the middle of the maze
                    # start = (start[0], start[1] + 0.5)
                    # end = (end[0], end[1] + 0.5)

                    # Append wall
                    walls.append((start, end))
        return walls

    def _get_edge(self,xy):
        # get edge of the wall given xy (array[],array[])
        edge=[]
        for k, (i, j) in enumerate(zip(xy[0], xy[1])):
            if i % 2 == 1:  # Vertical wall
                start = (i / 2, (j - 1) / 2)
                end = (i / 2, (j + 1) / 2)
            else:  # Horizontal wall
                start = ((i - 1) / 2, j / 2)
                end = ((i + 1) / 2, j / 2)
            edge.append((start,end))
        return edge

    def _get_maze_walls_near_turn(self, maze, wall_touch_path_0, wall_touch_path_1):
        """
        identify walls near turnpoints and return wall coordinate and indices
        indicies:
        UpRight:1, UpLeft:2, DownRight:3, DownLeft:4,
        """

        # deal with turnpoints
        kernel_ur = np.flipud(np.fliplr(np.array([[10, 0, 0, 0], [0, 0, 0, 0], [10, 0, 0, 0], [0, 10, 0, 10]])))
        kernel_ul = np.flipud(np.fliplr(np.array([[0, 0, 0, 10], [0, 0, 0, 0], [0, 0, 0, 10], [10, 0, 10, 0]])))
        kernel_dr = np.flipud(np.fliplr(np.array([[0, 10, 0, 10], [10, 0, 0, 0], [0, 0, 0, 0], [10, 0, 0, 0]])))
        kernel_dl = np.flipud(np.fliplr(np.array([[10, 0, 10, 0], [0, 0, 0, 10], [0, 0, 0, 0], [0, 0, 0, 10]])))

        conv_maze_ur = scipy_signal.convolve2d(
            maze,kernel_ur,mode='same', boundary='symm').astype(int)
        xy_ur = np.where(conv_maze_ur == 40)

        conv_maze_ul = scipy_signal.convolve2d(
            maze, kernel_ul, mode='same', boundary='symm').astype(int)
        xy_ul = np.where(conv_maze_ul == 40)

        conv_maze_dr = scipy_signal.convolve2d(
            maze, kernel_dr, mode='same', boundary='symm').astype(int)
        xy_dr = np.where(conv_maze_dr == 40)

        conv_maze_dl = scipy_signal.convolve2d(
            maze, kernel_dl, mode='same', boundary='symm').astype(int)
        xy_dl = np.where(conv_maze_dl == 40)

        # find actual wall near turnpoints
        xy_ur_u = (xy_ur[0], xy_ur[1]-2)
        xy_ur_r = (xy_ur[0]+1, xy_ur[1] - 1)

        xy_ul_u = (xy_ul[0], xy_ul[1]+1)
        xy_ul_l = (xy_ul[0] + 1, xy_ul[1])

        xy_dr_r = (xy_dr[0]-2, xy_dr[1] - 1)
        xy_dr_d = (xy_dr[0]-1, xy_dr[1] - 2)

        xy_dl_l = (xy_dl[0]-2, xy_dl[1])
        xy_dl_d = (xy_dl[0] - 1, xy_dl[1]+1)

        # get edge
        ur_u = self._get_edge(xy_ur_u)
        ur_r = self._get_edge(xy_ur_r)
        ul_u = self._get_edge(xy_ul_u)
        ul_l = self._get_edge(xy_ul_l)
        dr_r = self._get_edge(xy_dr_r)
        dr_d = self._get_edge(xy_dr_d)
        dl_l = self._get_edge(xy_dl_l)
        dl_d = self._get_edge(xy_dl_d)

        return ur_u,ur_r,ul_u,ul_l,dr_r,dr_d,dl_l,dl_d

    def _get_maze_walls_from_path(self, path):
        walls = []
        # from start/top
        for i, x in enumerate(path):
            if x[0] % 2==1: # Vertical wall
                start = (x[0] / 2, (x[1] - 1) / 2)
                end = (x[0] / 2, (x[1] + 1) / 2)
            else:
                start = ((x[0] - 1) / 2, x[1] / 2)
                end = ((x[0] + 1) / 2, x[1] / 2)

            walls.append((start, end))
        return walls

    def _num_turns_path(self, prey_path):
        path_x = prey_path[:, 0]
        path_x_diff = path_x[1:] - path_x[:-1]
        num_turns = np.sum(
            np.abs(np.convolve(path_x_diff, [-1, 1], mode='valid')))  # detect change point
        return num_turns

    def _prey_path_to_segment_length(self, prey_path):
        directions = np.array(prey_path[1:]) - np.array(prey_path[:-1])
        segments = [0]
        prev_direction = directions[0]
        for d in directions:
            if np.array_equal(d, prev_direction):
                segments[-1] = segments[-1] + 1
            else:
                segments.append(1)
            prev_direction = d

        return segments

    def _get_overlap_position(self, maze, path):
        x_overlap = []
        start_path = path[:-1]
        end_path = path[1:]

        # from start/top
        for i, (start,end) in enumerate(zip(start_path,end_path)):
            if start[0] % 2 == 0 and end[0] % 2 ==0: # vertical wall; note this is for original path before rotation
                left_wall = [np.int((start[0]+end[0])/2), start[1] - 1]
                right_wall = [np.int((start[0]+end[0])/2), start[1] + 1]
                if maze[left_wall[0],left_wall[1]] > 0 and maze[right_wall[0],right_wall[1]] > 0: # path crossing
                    x_overlap.append(i+1)
            if start[0] % 2 != 0 and end[0] % 2 != 0:  # horizontal wall
                up_wall = [start[0] - 1, np.int((start[1]+end[1])/2)]
                down_wall = [start[0] + 1, np.int((start[1]+end[1])/2)]
                if maze[up_wall[0],up_wall[1]] > 0 and maze[down_wall[0],down_wall[1]] > 0:  # path crossing
                    x_overlap.append(i + 1)
        return x_overlap

    def plot_maze(self,maze_walls,maze_prey_walls):
        for i,x in enumerate(maze_walls):
            start = x[0]
            end = x[1]
            plt.plot([start[0],end[0]],[start[1],end[1]],'k-')
        for i,x in enumerate(maze_prey_walls):
            start = x[0]
            end = x[1]
            plt.plot([start[0],end[0]],[start[1],end[1]],'.r-')
        plt.show()
        plt.axis('equal')

    def __call__(self):
        render_maze, original_maze, path, original_path, distractors = super(WireMazeSampler, self).__call__()

        # num_overlap = super(WireMazeSampler, self).num_overlap # do not take into account overlapping distractors
        x_overlap = self._get_overlap_position(original_maze,original_path)
        # print(x_overlap)

        # Because of python .imshow() weirdness, the top is the left side of the
        # screen, so have to rotate 270 degrees to correct for that
        maze, path = wire_path_dataset.rotate_maze_and_path_90(
            original_maze, path, num_times=3)
        original_path = wire_path_dataset.rotate_path_90(maze, original_path, num_times=3)
        # print(original_path)

        # transform distractor path same way
        for i, distractor_path in enumerate(distractors):
            distractors[i] = wire_path_dataset.rotate_path_90(maze, distractor_path, num_times=3)

        # deal with walls
        maze_width = int((maze.shape[0] + 1) / 2)
        maze_height = int((maze.shape[1] + 1) / 2)
        wall_touch_path_start, wall_touch_path_end = self._get_wall_touch_path(maze,original_path)
        maze_walls = self._get_maze_walls(maze,wall_touch_path_start, wall_touch_path_end)
        maze_walls = self._add_gap_distractor_junction(maze_walls)
        maze_prey_walls = self._get_maze_walls_from_path(original_path)

        maze_walls_near_turn = self._get_maze_walls_near_turn(maze,wall_touch_path_start, wall_touch_path_end)

        # self.plot_maze(maze_walls,maze_prey_walls)

        # original
        # prey_path = [tuple(x) for x in (path[::2] / 2).astype(int)]

        # NW: I'm not sure what this line does --- might no longer be correct
        # HS: I think this is for undoing (_augment_path) transformation into pixels (2 for pixels_per_square)
        prey_path = [x for x in (path[::2] / 2).astype(int)] # 0 to 14

        # # Might need some lines like this to extend the path
        # prey_path.append(prey_path[-1] + (prey_path[-1] - prey_path[-2]))
        prey_path.insert(0, prey_path[0] + (prey_path[0] - prey_path[1]))

        prey_path = [tuple(x) for x in prey_path]
        # print(prey_path)
        # if prey_path[0][1]==6:
        #     print(prey_path)

        segment_length = self._prey_path_to_segment_length(prey_path)
        # print(segment_length)

        num_turns = self._num_turns_path(path)

        features = {
            'name': 'WireMaze',
            'start_x': prey_path[0][1],
            'num_turns': num_turns,
            'path_length': len(prey_path),
            'num_overlap': len(x_overlap),
            'x_overlap' : x_overlap,
            'maze_prey_walls': maze_prey_walls,
            'distractor_path':distractors,
            # 'maze_walls_near_turn': maze_walls_near_turn,
        }

        stimulus = dict(
            maze_width=maze_width,
            maze_height=maze_height,
            prey_path=prey_path,
            maze_walls=maze_walls,
            features=features,
        )

        return stimulus

    def __len__(self):
        return self._num_mazes  # np.inf
