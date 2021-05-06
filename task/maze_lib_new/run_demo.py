"""Demo maze random generation."""

from absl import app
import collections
import maze as maze_lib
from moog import observers
from matplotlib import pyplot as plt

_WIDTH = 10
_HEIGHT = 10

_PREY_PATH = [
    (2, 0), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (5, 3), (6, 3), (6, 2),
    (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (6, 7), (5, 7), (5, 8),
    (5, 9),
]


def main(_):
    """Run demo.
    TODO: implement _PREY_PATH (# turns, path length) -> collect pilot for RT

    """

    # Generate the maze
    maze = maze_lib.Maze(width=_WIDTH, height=_HEIGHT, prey_path=_PREY_PATH)
    maze.sample_distractors()

    # Render the maze
    wall_sprite_factors = dict(c0=180, c1=0., c2=0.5) # gray
    wall_sprites = maze.to_sprites(wall_width=0.1, **wall_sprite_factors)
    state = collections.OrderedDict([
        ('walls', wall_sprites)
    ])

    renderer = observers.PILRenderer(
        image_size=(256, 256),
        color_to_rgb='hsv_to_rgb',
    )
    image = renderer(state)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    app.run(main)
