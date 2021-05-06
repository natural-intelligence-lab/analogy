"""Conditions."""

import numpy as np

_width = 10  # num of cells of grid; should match maze_lib_new/maze.Maze input
_height = 10

_min_length_range = 8  # in unit of cells
_max_length_range = 25

_min_num_turns_range = 1
_max_num_turns_range = 3

_min_arm_length = 3

class Base():
    """generate prey path
    random sampling of # turns and path length
    rejection sampling if
        1) path doesn't fit in grid (_w, _h),
        2) segment length < _min_arm_length

    algorithm forces the following conditions
        1) side of end point is different from side of entry point (avoiding loop; complicating eye movement & leading to popup effect)
        2) after sampling direction & arm length, move around path within grid to make entry and end point touch sides

    """

    _NUM_SAMPLES_PER_NUM_TURNS = 100

    def __init__(self,
                 length_range,
                 num_turns_range,
                 name=''):
        self._length_range = length_range
        self._num_turns_range = num_turns_range
        self._min_arm_length = _min_arm_length
        self._name = name

    def _random_num_turns(self):
        return np.random.randint(
            self._num_turns_range[0], self._num_turns_range[1])

    def _sample_vertices(self, num_turns, length):
        turn_points = np.random.randint(0, length, size=(num_turns,))
        turn_points = np.sort(turn_points)
        vertices = np.concatenate(([0], turn_points, [length]))
        pairwise_distances = np.abs(
            vertices[:, np.newaxis] - vertices[np.newaxis, :])
        np.fill_diagonal(pairwise_distances, np.inf)
        length_1 = np.sum(pairwise_distances[::2]) - (np.size(pairwise_distances[::2]) - 1)  # remove overlapped cell
        length_2 = np.sum(pairwise_distances[1::2]) - (np.size(pairwise_distances[1::2]) - 1)
        if np.sum(pairwise_distances < self._min_arm_length) > 0:  # rejection sampling
            return self._sample_vertices(num_turns, length)
        elf length_1 < _width:  # path doesn't fit in grid (_w, _h),
            return self._sample_vertices(num_turns, length)
        elf length_2 > _height:
            return self._sample_vertices(num_turns, length)
        else:
            return vertices

    def _sample_arm(self, num_turns, length, start_direction):
        vertices = self._sample_vertices(num_turns, length)
        lengths = vertices[1:] - vertices[:-1]

        # direction: choose horizontal if vertical (vice versa)
        directions = [start_direction]
        # initialize length_x length_y
        if directions[-1] == 'r' or directions[-1] == 'l':
            length_x = lengths[0]
            length_y = [0]
        if directions[-1] == 'u' or directions[-1] == 'd':
            length_x = [0]
            length_y = lengths[0]
        for i in range(1, num_turns):
            if directions[-1] == 'r' or directions[-1] == 'l':
                directions.append(np.random.choice(['u', 'd']))
                length_y += lengths[i]
            if directions[-1] == 'u' or directions[-1] == 'd':
                directions.append(np.random.choice(['r', 'l']))
                length_x += lengths[i]

        # after sampling direction & arm length, move around path within grid to make entry and end point touch sides
        # _END = (0.5, 0.1)

        # first sum up hor/ver length
        odd = lengths[::2]

        even = lengths[1::2]



        arm = {
            'end': Base._END,
            'directions': ''.join(directions),
            'lengths': lengths.tolist(),
        }

        # convert arm into list of cell index

        arm_full

        return arm, arm_full

    def _sample_condition(self, num_turns):
        prey_arm_length = np.random.randint(
            self._length_range[0], self._length_range[1])

        prey_dir = np.random.choice(['l', 'r', 'u', 'd'])  # initial directions

        prey_arm, prey_full_arm = self._sample_arm(
            num_turns, length=prey_arm_length, start_direction=prey_dir)

        features = {
            'name': self._name,
            'prey_arm_turns': num_turns,
            'prey_arm_length': prey_arm_length,
        }
        condition = [[prey_arm, prey_full_arm], 0, features,]
        return condition

    def _conditions_per_num_turns(self, num_turns):
        conditions = [
            self._sample_condition(num_turns)
            for _ in range(Base._NUM_SAMPLES_PER_NUM_TURNS)
        ]
        return conditions

    def __call__(self):
        conditions = []
        for num_turns in range(
                self._num_turns_range[0], self._num_turns_range[1]):
            conditions.extend(self._conditions_per_num_turns(num_turns))
        return conditions


class Random3(Base):
    """Random with 1 to 3 prey turns."""

    def __init__(self):
        length_range = [_min_length_range, _max_length_range+1]  # upper: exclusive
        num_turns_range = [_min_num_turns_range, _min_num_turns_range+1]  # upper: exclusive
        super(Random3, self).__init__(
            length_range, num_turns_range, name='Random3')