"""Demo maze sampling."""

from absl import app
from matplotlib import pyplot as plt
import maze_composer
import numpy as np
import os

_NUM_LAYERS = 4  # Number of paths to layer to make a maze
_SQRT_SAMPLES = 4  # Square root of the number of maze samples to display
_PIXELS_PER_SQUARE = 6  # Pixels per maze grid square. Should be an even number.
_RENDER_PATH = True  # Whether to render the ball path in gray.
_BALL_PATH_TOP_BOTTOM = True
_MAX_NUM_TURNS = 5


def main(_):
    path_dir = os.path.join(
        os.getcwd(), 'path_datasets', 'maze_size_16',
        'samples_per_pair_2_v0')
    composer = maze_composer.MazeComposer(
        path_dir=path_dir,
        num_layers=_NUM_LAYERS,
        pixels_per_square=_PIXELS_PER_SQUARE,
        ball_path_top_bottom=_BALL_PATH_TOP_BOTTOM,
        max_num_turns=_MAX_NUM_TURNS,
    )

    fig, axes = plt.subplots(_SQRT_SAMPLES, _SQRT_SAMPLES, figsize=(9, 9))
    for i in range(_SQRT_SAMPLES):
        for j in range(_SQRT_SAMPLES):
            maze, _, path = composer()
            if _RENDER_PATH:
                maze[path[:, 0], path[:, 1]] = 0.5
            axes[i, j].imshow(maze, cmap='gray')
            axes[i, j].axis('off')
    print(f'Maze shape: {maze.shape}')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    app.run(main)
