"""Run demo.

This script can be used to play with and test prototype tasks with a
keyboard/mouse interface.
"""

from absl import app
from absl import flags
import importlib
import os

from moog import observers
from moog_demos import gif_writer as gif_writer_lib

import sys
sys.path.append('..')

from moog import environment
from configs import levels
from python_demo import human_agent
from utils import logger_env_wrapper

FLAGS = flags.FLAGS
flags.DEFINE_string('level',
                    'random.random_4.random_4',
                    'Level.')
flags.DEFINE_integer('render_size', 512,
                     'Height and width of the output image.')
flags.DEFINE_integer('anti_aliasing', 1, 'Renderer anti-aliasing factor.')
flags.DEFINE_integer('fps', 20, 'Frames per second.')
flags.DEFINE_integer('reward_history', 30,
                     'Number of historical reward timesteps to plot.')

# Flags for gif writing
flags.DEFINE_boolean('write_gif', True, 'Whether to write a gif.')
flags.DEFINE_string('gif_file',
                    os.path.join(os.getcwd(), 'logs/gifs/test_0.gif'),
                    'File path to write the gif to.')
flags.DEFINE_integer('gif_fps', 15, 'Frames per second for the gif.')


def main(_):
    """Run interactive task demo."""

    level_split = FLAGS.level.split('.')
    config_module = __import__(
        '.'.join(['configs', 'levels'] + level_split[:-1]),
        globals(),
        locals(),
        [level_split[-2]],
    )
    config_class = getattr(config_module, level_split[-1])
    
    config_instance = config_class(
        fixation_phase=False,
        delay_phase=False,
        ms_per_unit=800,
    )

    config = config_instance()
    config['observers']['image'] = observers.PILRenderer(
        image_size=(FLAGS.render_size, FLAGS.render_size),
        anti_aliasing=FLAGS.anti_aliasing,
        color_to_rgb=config['observers']['image'].color_to_rgb,
    )
    env = environment.Environment(**config)
    # env = logger_env_wrapper.MazePongLoggingEnvironment(env)

    if FLAGS.write_gif:
        gif_writer = gif_writer_lib.GifWriter(
            gif_file=FLAGS.gif_file,
            fps=FLAGS.gif_fps,
        )
    else:
        gif_writer = None
    
    # Constructing the agent automatically starts the environment
    human_agent.HumanAgent(
        env,
        render_size=FLAGS.render_size,
        fps=FLAGS.fps,
        reward_history=FLAGS.reward_history,
        gif_writer=gif_writer,
    )


if __name__ == "__main__":
    app.run(main)
