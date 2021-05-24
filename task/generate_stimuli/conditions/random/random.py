"""Conditions."""

import numpy as np
import maze_lib

_MAX_TRIES = int(1e4)
_MIN_SEGMENT_LENGTH = 3

_DIRECTIONS_NAMED = {
    'N': (0, 1),
    'S': (0, -1),
    'E': (1, 0),
    'W': (-1, 0),
}
_DIRECTIONS = [np.array(x) for x in list(_DIRECTIONS_NAMED.values())]


class PreyPathGenerator():
    """Random prey path.
        exit always at South side, randomly selected from uniform distribution
        entry among North, East, West but South not allowed
        no control over # turns, total path length
    """

    def __init__(self, maze_size, min_segment_length):
        self._maze_size = maze_size
        self._min_segment_length = min_segment_length
    
    def __call__(self, num_tries=0):
        if num_tries > _MAX_TRIES:
            raise ValueError('Could not generate a prey path.')
        
        maze_array, prey_path = self._sample_exit()
        finished = False
        while not finished:
            # Sample segment
            exists_valid_segment = self._add_segment(maze_array, prey_path)
            if not exists_valid_segment:
                return self(num_tries=num_tries + 1)
            
            # Check if should finish
            tail = prey_path[-1]
            d = prey_path[-1] - prey_path[-2]
            if tail[0] == 0 and tuple(d) == _DIRECTIONS_NAMED['W']:
                finished = True
            elif (tail[0] == self._maze_size - 1 and
                    tuple(d) == _DIRECTIONS_NAMED['E']):
                finished = True
            elif (tail[1] == self._maze_size - 1 and
                    tuple(d) == _DIRECTIONS_NAMED['N']):
                finished = True
            elif tail[1] == 0 and tuple(d) == _DIRECTIONS_NAMED['S']:
                # Do not allow prey to enter from bottom
                return self(num_tries=num_tries + 1)

        # Reverse prey path because we built it back-to-front
        prey_path = prey_path[::-1]

        return prey_path

    def _valid_lengths(self, maze_array, prey_path, direction):
        tail = prey_path[-1]
        lengths = [0]
        for i in range(self._maze_size):
            tail = tail + direction
            if np.any(tail < 0) or np.any(tail == self._maze_size): # get out of maze
                break
            if maze_array[tail[0], tail[1]]: # already exists in maze
                break
            lengths.append(lengths[-1] + 1)

        lengths = [l for l in lengths if l >= self._min_segment_length]
        return lengths

    def _valid_segments(self, maze_array, prey_path):
        segments = []
        for direction in _DIRECTIONS:
            valid_lengths = self._valid_lengths(
                maze_array, prey_path, direction)
            segments.extend([(direction, l) for l in valid_lengths])
            
        return segments

    def _add_segment(self, maze_array, prey_path):
        valid_segments = self._valid_segments(maze_array, prey_path)
        if len(valid_segments) == 0:
            return False
        else:
            segment = valid_segments[np.random.choice(len(valid_segments))]
            direction, length = segment
            
            # Add segment
            tail = prey_path[-1]
            for i in range(length):
                tail = tail + direction
                prey_path.append(tail)
                maze_array[tail[0], tail[1]] = 1

            return True

    def _sample_exit(self):
        maze_array = np.zeros((self._maze_size, self._maze_size))
        exit_x = int(np.random.randint(0, self._maze_size))
        prey_path = []
        for i in range(self._min_segment_length):
            maze_array[exit_x, i] = 1
            prey_path.append(np.array([exit_x, i]))
        
        return maze_array, prey_path


class Random12():

    _NUM_CONDITIONS = int(1e3)
    _MAZE_SIZE = 12

    def __init__(self):
        self._maze_size = Random12._MAZE_SIZE
        self._prey_path_generator = PreyPathGenerator(
            maze_size=self._maze_size, min_segment_length=_MIN_SEGMENT_LENGTH)

    def _prey_path_to_segments(self, prey_path):
        directions = np.array(prey_path[1:]) - np.array(prey_path[:-1])
        segments = [[prey_path[1]-prey_path[0]]]  # debug 2021/5/8
        prev_direction = directions[0]
        for d in directions[1:]:
            if np.array_equal(d, prev_direction):
                segments[-1].append(d)
            else:
                segments.append([d])
            prev_direction = d

        return segments

    def _sample_condition(self):
        prey_path = self._prey_path_generator()
        maze = maze_lib.Maze(
            width=self._maze_size, height=self._maze_size, prey_path=prey_path)
        maze.sample_distractor_exit(prey_path=prey_path)
        maze.sample_distractors()
        maze_walls = maze.walls

        start_x = prey_path[0][0]
        segments = self._prey_path_to_segments(prey_path)
        num_turns = len(segments) - 1
        path_length = sum(len(x) for x in segments)

        features = {
            'name': 'Random12',
            'start_x': start_x,
            'num_turns': num_turns,
            'path_length': path_length,
        }
        maze_width = self._maze_size
        maze_height = self._maze_size
        condition = [maze_width, maze_height, prey_path, maze_walls, features]
        return condition

    def __call__(self):
        conditions = [
            self._sample_condition()
            for _ in range(Random12._NUM_CONDITIONS)
        ]
        return conditions
