"""Conditions."""

import copy
import logging
import maze_lib
import numpy as np


class BaseRandom():

    def __init__(self,
                 features,
                 num_arm_segments,
                 separation_threshold=2,
                 num_samples=1000):
        self._features = features
        self._num_arm_segments = num_arm_segments
        self._num_samples = num_samples
        self._separation_threshold = separation_threshold

    @property
    def allow_continuation(self):
        return True

    def _generate_condition(self):

        # Generate maze
        maze = maze_lib.Maze(separation_threshold=self._separation_threshold)
        for i in range(4):
            num_segments = np.random.choice(self._num_arm_segments)
            valid_arm = maze.sample_arm(
                maze_lib.Directions[i],
                num_segments,
                allow_continuation=self.allow_continuation,
            )
            if not valid_arm:
                return self._generate_condition()

        # Generate prey
        prey_arm = np.random.randint(4)

        f = copy.deepcopy(self._features)
        f['prey_arm'] = prey_arm

        return maze.arms, prey_arm, f

    def __call__(self):
        conditions = []
        n_samples = self._num_samples
        for i in range(n_samples):
            if i % 10 == 0:
                logging.info(f'Generated:  {i} / {n_samples}')
            conditions.append(self._generate_condition())

        return conditions


class Random2(BaseRandom):

    def __init__(self):
        super(Random2, self).__init__(
            features={'name': 'Random2'},
            num_arm_segments=[2],
        )


class Random3(BaseRandom):

    def __init__(self):
        super(Random3, self).__init__(
            features={'name': 'Random3'},
            num_arm_segments=[3],
            num_samples=10,
        )


class Random4(BaseRandom):

    def __init__(self):
        super(Random4, self).__init__(
            features={'name': 'Random4'},
            num_arm_segments=[4],
            num_samples=10,
        )