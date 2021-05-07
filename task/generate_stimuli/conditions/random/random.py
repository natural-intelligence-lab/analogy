"""Conditions."""

import numpy as np

_MAX_TRIES = int(1e4)
_MIN_SEGMENT_LENGTH = 2

_DIRECTIONS_NAMED = {
    'N': (0, 1),
    'S': (0, -1),
    'E': (1, 0),
    'W': (-1, 0),
}
_DIRECTIONS = [np.array(x) for x in list(_DIRECTIONS_NAMED.values())]


class PreyPathGenerator():
    """Random prey path."""

    def __init__(self, maze_size, min_segment_length):
        self._maze_size = maze_size
        self._min_segment_length = min_segment_length
    
    def __call__(self, num_tries=0):
        if num_tries > _MAX_TRIES:
            raise ValueError('Could not generate a prey path.')
        
        maze_array, prey_path = self._sample_start()
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
            elif tail[0] == 0 and tuple(d) == _DIRECTIONS_NAMED['S']:
                finished = True

        return maze_array, prey_path

    def _valid_lengths(self, maze_array, prey_path, direction):
        tail = prey_path[-1]
        lengths = [0]
        for i in range(self._maze_size):
            tail = tail + direction
            if np.any(tail < 0) or np.any(tail == self._maze_size):
                break
            if maze_array[tail[0], tail[1]]:
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

    def _sample_start(self):
        maze_array = np.zeros((self._maze_size, self._maze_size))
        start_x = int(np.random.randint(0, self._maze_size))
        prey_path = []
        for i in range(self._min_segment_length):
            maze_array[start_x, self._maze_size - 1 - i] = 1
            prey_path.append(np.array([start_x, self._maze_size - 1 - i]))
        
        return maze_array, prey_path


class Random12():

    _NUM_CONDITIONS = int(1e3)

    def __init__(self):
        self._maze_size = 12
        self._prey_path_generator = PreyPathGenerator(
            maze_size=self._maze_size, min_segment_length=_MIN_SEGMENT_LENGTH)

    def _prey_path_to_segments(self, prey_path):
        directions = np.array(prey_path[1:]) - np.array(prey_path[:-1])
        segments = [[]]
        prev_direction = directions[0]
        for d in directions[1:]:
            if np.array_equal(d, prev_direction):
                segments[-1].append(d)
            else:
                segments.append([d])
            prev_direction = d

        return segments

    def _sample_condition(self):
        maze_array, prey_path = self._prey_path_generator()

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
        condition = [self._maze_size, prey_path, features]
        return condition

    def __call__(self):
        conditions = [
            self._sample_condition()
            for _ in range(Random12._NUM_CONDITIONS)
        ]
        return conditions
