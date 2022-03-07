"""MAze composer class."""

import pdb
import numpy as np
import os
from wire_maze_lib import path_dataset
# import path_dataset
from scipy import signal as scipy_signal

_MAX_TRIES = int(1e4)


class MazeComposer():
    """Generates random mazes composed of overlaying paths."""

    def __init__(self,
                 path_dir,
                 num_layers,
                 pixels_per_square=6,
                 ball_path_top_bottom=True,
                 distractors_top_bottom=True,
                 max_num_turns=np.inf):
        """Constructor.
        
        Args:
            path_dir: String. Directory of path dataset to use for composing
                mazes.
            num_layers: Int. Number of paths to compose for each maze.
            pixels_per_square: Int. Number of pixels for each maze square during
                rendering.
            ball_path_top_bottom: Bool. Whether the ball path should be forced
                to enter from the top and exit from the bottom.
            distractors_top_bottom: Bool. Whether all distractor paths should be
                forced to enter from the top and exit from the bottom.
            max_num_turns: Int. Maximum number of turns for the ball path.
        """
        self._num_layers = num_layers
        self._pixels_per_square = pixels_per_square
        self._ball_path_top_bottom = ball_path_top_bottom
        self._distractors_top_bottom = distractors_top_bottom
        self._max_num_turns = max_num_turns

        if pixels_per_square % 2 != 0:
            raise ValueError(
                f'pixels_per_square is {pixels_per_square} but must be even '
                'for the ball path to line up in the center of the maze path.')

        # Load mazes
        filenames = os.listdir(path_dir)
        self._mazes = []
        for k in filenames:
            new_mazes = np.load(os.path.join(path_dir, k), allow_pickle=True)
            self._mazes.extend(new_mazes)
        self._num_mazes = len(self._mazes)

        # Get valid ball path indices
        self._valid_ball_path_inds = [
            i for i, maze in enumerate(self._mazes)
            if self._valid_ball_path(maze)
        ]

    def _valid_ball_path(self, maze):
        """Check if maze is a valid ball path."""
        if self._ball_path_top_bottom:
            start_top = maze[1][0][0] == 0
            end_bottom = maze[1][-1][0] == maze[0].shape[0] - 1
            if not (start_top and end_bottom):
                return False
        if np.isfinite(self._max_num_turns):
            num_turns = self._num_turns(maze)
            if num_turns > self._max_num_turns:
                return False
        return True

    def _num_turns(self, maze):
        """Get number of turns in a maze."""
        return np.sum(maze[0] == -1)

    def _transform_maze(self, maze, path):
        """Randomly flip/rotate maze and path."""
        if np.random.rand() < 0.5:
            maze, path = path_dataset.flip_maze_and_path_up_down(
                maze, path)
        num_rotations = np.random.randint(4)
        maze, path = path_dataset.rotate_maze_and_path_90(
            maze, path, num_times=num_rotations)
        return maze, path

    def _render_maze(self, maze):
        """Convert small binary maze array to full-size maze array.
        
        The returned maze array has values 0 for outside the path and 1 for path
        interior. This coding is important for the maze composition.
        """
        p = self._pixels_per_square
        p2 = int(p / 2)
        maze_size = (np.array(maze.shape) + 1) / 2
        maze_size = p * maze_size
        render = np.zeros(tuple(maze_size.astype(int)))
        for i, row in enumerate(maze):
            for j, x in enumerate(row):
                if x == 1:
                    if i % 2 == 0:
                        ind_i = (i / 2) * p
                        ind_j = ((j - 1) / 2) * p + p2
                        horizontal = False
                    else:
                        ind_i = ((i - 1) / 2) * p + p2
                        ind_j = (j / 2) * p
                        horizontal = True
                    ind_i = int(ind_i)
                    ind_j = int(ind_j)
                    w = p2 + 1
                    if horizontal:
                        render[ind_i - 1 : ind_i + 1, ind_j - w : ind_j + w] = 1
                    else:
                        render[ind_i - w : ind_i + w, ind_j - 1 : ind_j + 1] = 1
        return render

    def _overlay_mazes(self, mazes):
        """Overlay a list of mazes with occlusion, in order."""
        final_maze = mazes[0]
        for maze in mazes[1:]:
            final_maze *= (maze == 0)
            final_maze += maze
        final_maze[final_maze < 0] = 0.
        return final_maze

    def _augment_path(self, path):
        """Convert path in units of grid square to units of pixels.
        
        The returned path is self._pixels_per_square times longer than the input
        path.
        """
        path = np.repeat(path, self._pixels_per_square, axis=0)
        kernel = np.ones((self._pixels_per_square, 1))
        kernel /= self._pixels_per_square
        path = scipy_signal.convolve2d(
            path, kernel, mode='valid', boundary='symm').astype(int)
        return path

    def __call__(self):
        """Sample a maze."""

        # Get ball path
        maze, path = self._mazes[
            np.random.choice(self._valid_ball_path_inds)]
        maze = np.copy(maze)
        path = np.copy(path)
        path = (0.25 * (path[1:] + path[:-1])).astype(int)
        path *= self._pixels_per_square
        path += int(self._pixels_per_square / 2)
        path = self._augment_path(path)

        # Add distractor paths
        for _ in range(self._num_layers - 1):
            done = False
            count  = 0
            while not done:
                if count > _MAX_TRIES:
                    raise ValueError('Could not generate a prey path.')
                if self._distractors_top_bottom:
                    new_maze, _ = self._mazes[
                        np.random.choice(self._valid_ball_path_inds)]
                else:
                    new_maze, _ = self._transform_maze(*self._mazes[
                        np.random.randint(self._num_mazes)])
                if not np.any(new_maze * maze != 0):
                    maze += new_maze
                    done = True

        rendered_maze = self._render_maze(maze)

        return rendered_maze, maze, path
