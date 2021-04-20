"""Conditions."""

import numpy as np


class Base():
    """Single handcrafted maze."""

    _NUM_SAMPLES_PER_NUM_TURNS = 10
    _END = (0.5, 0.1)

    def __init__(self,
                 length_range,
                 num_turns_range,
                 name='',
                 min_arm_length=0.1):
        self._length_range = length_range
        self._num_turns_range = num_turns_range
        self._min_arm_length = min_arm_length
        self._name = name

    def _random_num_turns(self):
        return np.random.randint(
            self._num_turns_range[0], self._num_turns_range[1])

    def _sample_vertices(self, num_turns, length):
        turn_points = np.random.uniform(0, length, size=(num_turns,))
        turn_points = np.sort(turn_points)
        vertices = np.concatenate(([0], turn_points, [length]))
        pairwise_distances = np.abs(
            vertices[:, np.newaxis] - vertices[np.newaxis, :])
        np.fill_diagonal(pairwise_distances, np.inf)
        if np.sum(pairwise_distances < self._min_arm_length) > 0:
            return self._sample_vertices(num_turns, length)
        else:
            return vertices

    def _sample_arm(self, num_turns, length, start_direction):
        vertices = self._sample_vertices(num_turns, length)
        directions = [start_direction, 'u']
        for i in range(1, num_turns):
            if i % 2 == 0:
                directions.append('u')
            else:
                directions.append(np.random.choice(['l', 'r']))

        lengths = vertices[1:] - vertices[:-1]
        arm = {
            'end': Base._END,
            'directions': ''.join(directions),
            'lengths': lengths.tolist(),
        }

        return arm

    def _sample_condition(self, num_turns):
        prey_arm_length = np.random.uniform(
            self._length_range[0], self._length_range[1])
        distractor_arm_length = np.random.uniform(
            self._length_range[0], self._length_range[1])
        
        if np.random.rand() < 0.5:
            prey_dir = 'l'
            distractor_dir = 'r'
        else:
            prey_dir = 'r'
            distractor_dir = 'l'

        prey_arm = self._sample_arm(
            num_turns, length=prey_arm_length, start_direction=prey_dir)
        
        distractor_num_turns = self._random_num_turns()
        distractor_arm = self._sample_arm(
            distractor_num_turns, length=distractor_arm_length,
            start_direction=distractor_dir)

        features = {
            'name': self._name,
            'prey_arm_turns': num_turns,
            'prey_arm_length': prey_arm_length,
            'distractor_arm_turns': distractor_num_turns,
            'distractor_arm_length': distractor_arm_length,
        }
        condition = [[prey_arm, distractor_arm], 0, features,]
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
        length_range = [0.5, 0.8]
        num_turns_range = [1, 4]
        super(Random3, self).__init__(
            length_range, num_turns_range, name='Random3')