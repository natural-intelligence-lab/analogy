"""Plot an image of a stimulus condition."""

from absl import app
from absl import flags
from matplotlib import pyplot as plt
from moog import environment
from moog import observers
import numpy as np

import sys
sys.path.append('..')

from maze_pong_configs import config as config_lib
import maze_lib


# _MAZE_SCHEMA = [
#     '  |_       ',
#     '   _|_ _   ',
#     '  |_   _|  ',
#     '    |_|_ _ ',
#     '      |   |',
# ]

_MAZE_SCHEMA = [
    '   _|_   ',
    ' _|   |_ ',
    '| |   | |',
    '| |   | |',
    '| |   | |',
]
_MAZE_SCHEMA = [
    '   _|_   ',
    ' _|  _|_ ',
    '| | |   |',
    '| | |   |',
    '| | |   |',
]
_PREY = [2]

_MAZE_SCHEMA = [
    '  |     |  ',
    ' _|_   _|_ ',
    '|   | |   |',
    '|   | |   |',
    '|   | |   |',
]
_PREY = [1, 4]
_MAZE_SCHEMA = [
    '  |   |  ',
    ' _|_  |  ',
    '|   |_|_ ',
    '|   |   |',
    '|   |   |',
]
_PREY = [1, 3]


class Sampler():
    def __init__(self):
        """Constructor."""
        rollouts = [((0, p), 1) for p in _PREY]
        self._stimulus = dict(
            maze_matrix=maze_lib.Maze.from_schema(_MAZE_SCHEMA).maze_matrix,
            prey=_PREY,
            paths=rollouts,
            features={},
        )

    def __call__(self):
        return self._stimulus


def main(_):
    """Run interactive task demo."""

    stimulus_generator = Sampler()
    config = config_lib.Config(stimulus_generator, fixation_phase=False)()

    config['observers']['image'] = observers.PILRenderer(
        image_size=(512, 512),
        anti_aliasing=2,
        color_to_rgb=config['observers']['image'].color_to_rgb,
    )
    env = environment.Environment(**config)
    env.reset()

    while env.meta_state['phase'] != 'planning':
        env.step({'eye': 0.5 * np.ones(2), 'controller': np.zeros(2)})

    env.step({'eye': 0.5 * np.ones(2), 'controller': np.zeros(2)})
    obs = env.observation()

    plt.figure()
    plt.imshow(obs['image'])
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    app.run(main)
