"""Run stimuli generation."""

from absl import app
import importlib
import shutil
import os

import sys
sys.path.append('..')

import conditions
import generator

# _CONDITION = 'random.Random12Square'
_CONDITION = 'random.VerticalPreyRandomHeight'
# _CONDITION = 'training.VerticalTiming'

# Log directory
_LOGDIR = os.path.join(os.getcwd(), '../stimuli')
# Can be used to distinguish multiple datasets of the same condition
_LOGDIR_SUFFIX = ''
# How often to print a progress report
_LOG_EVERY = 10


def main(_):
    """Run interactive task demo."""
    condition_split = _CONDITION.split('.')
    log_dir = os.path.join(
        _LOGDIR,
        *condition_split[:-1],
        condition_split[-1] + _LOGDIR_SUFFIX,
    )

    # If log_dir exists, ask user whether to overwrite
    if os.path.exists(log_dir):
        print(f'Directory {log_dir} to stimuli already exists.'.format(log_dir))
        should_override = input(
            'Would you like to overwrite the data there?  (y/n)')
        if should_override == 'y':
            print('Removing {}'.format(log_dir))
            shutil.rmtree(log_dir)
        else:
            print('exiting')
            sys.exit()
    os.makedirs(log_dir)

    # Generate stimuli
    condition_module = '.'.join(['conditions'] + condition_split[:-1])
    condition_module = importlib.import_module(condition_module)
    condition_class = getattr(condition_module, condition_split[-1])()
    generator.generate_stimuli(
        condition_class(),
        log_dir,
        log_every=_LOG_EVERY,
    )


if __name__ == "__main__":
    app.run(main)
