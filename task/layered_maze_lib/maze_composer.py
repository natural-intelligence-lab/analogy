"""MAze composer class."""

import numpy as np
import os
from layered_maze_lib import path_dataset
from scipy import signal as scipy_signal


class MazeComposer():
    """Generates random mazes composed of overlaying paths."""

    def __init__(self,
                 path_dir,
                 num_layers,
                 pixels_per_square=4,
                 ball_path_top_bottom=True,
                 ball_path_top=True,
                 max_num_turns=np.inf,
                 num_turns=None):
        """Constructor.
        
        Args:
            path_dir: String. Directory of path dataset to use for composing
                mazes.
            num_layers: Int. Number of paths to compose for each maze.
            pixels_per_square: Int. Number of pixels for each maze square during
                rendering.
            ball_path_top_bottom: Bool. Whether the ball path should be forced
                to enter from the top and exit from the bottom.
            max_num_turns: Int. Maximum number of turns for the ball path.
            num_turns: Int. if not None, number of turns for the ball path.
        """
        self._num_layers = num_layers
        self._pixels_per_square = pixels_per_square
        self._ball_path_top_bottom = ball_path_top_bottom
        self._ball_path_top = ball_path_top
        self._max_num_turns = max_num_turns
        self._num_turns_specified = num_turns

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
        num_turns = self._num_turns(maze)
        if self._ball_path_top_bottom:
            start_top = maze[1][0][0] == 0
            end_bottom = maze[1][-1][0] == maze[0].shape[0] - 1
            if not (start_top and end_bottom):
                return False
        if self._ball_path_top:
            start_top = maze[1][0][0] == 0
            if not start_top:
                return False

        if np.isfinite(self._max_num_turns):
            if num_turns > self._max_num_turns:
                return False
        if self._num_turns is not None: # num_turns specified
            if num_turns != self._num_turns_specified:
                return False
        return True

    def _num_turns(self, maze):
        """Get number of turns in a maze."""
        path_x = maze[1][:, 0]
        path_x_diff = path_x[1:] - path_x[:-1]
        num_turns = np.sum(
            np.abs(np.convolve(path_x_diff, [-1, 1], mode='valid')))
        return num_turns

    def _transform_maze(self, maze, path):
        """Randomly flip/rotate maze and path."""
        if np.random.rand() < 0.5:
            maze, path = path_dataset.flip_maze_and_path_up_down(maze, path)
        num_rotations = np.random.randint(4)
        maze, path = path_dataset.rotate_maze_and_path_90(
            maze, path, num_times=num_rotations)
        return maze, path

    def _render_maze(self, maze):
        """Convert small binary maze array to full-size maze array.
        
        The returned maze array has values 0 for outside the path, 1 for path
        walls, and -1 for path interior. This coding is important for the maze
        composition.
        """
        maze = np.repeat(
            np.repeat(maze, self._pixels_per_square, axis=0),
            self._pixels_per_square, axis=1)
        kernel = np.ones((2, 2))
        maze = scipy_signal.convolve2d(
            maze, kernel, mode='valid', boundary='symm')
        maze[maze == 4] = -1
        maze[maze > 0] = 1
        return maze

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

    def _sample_ball_path(self):
        return self._mazes[np.random.choice(self._valid_ball_path_inds)]

    def __call__(self):
        """Sample a maze."""
        mazes = [
            self._mazes[np.random.randint(self._num_mazes)]
            for _ in range(self._num_layers - 1)
        ]
        mazes = [self._transform_maze(*x) for x in mazes]
        mazes.append(self._sample_ball_path())
        
        path = mazes[-1][1]
        path = self._pixels_per_square * path
        path += int(self._pixels_per_square / 2) - 1
        path = self._augment_path(path)
        mazes = [x[0] for x in mazes]
        mazes = [self._render_maze(x) for x in mazes]
        maze = self._overlay_mazes(mazes)
        return maze, path
