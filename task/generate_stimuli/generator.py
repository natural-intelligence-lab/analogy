"""Stimuli generator class."""

import json
import logging
import numpy as np
import os

# This is the number of numerals in filenames. Since there is one file per
# condition, you should pick _FILENAME_ZFILL large enough that the number of
# condition in your stimulus dataset is less than 10^_FILENAME_ZFILL.
_FILENAME_ZFILL = 5 # 4


def _serialize(x):
    """Serialize a value x."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, (np.float32, np.float64)):
        return float(x)
    elif isinstance(x, (np.int32, np.int64)):
        return int(x)
    elif isinstance(x, list):
        return list([_serialize(a) for a in x])
    elif isinstance(x, tuple):
        return tuple(_serialize(a) for a in x)
    elif isinstance(x, dict):
        return {k: _serialize(v) for k, v in x.items()}
    else:
        return x


def generate_stimuli(conditions, log_dir, log_every=1):

    num_conditions = len(conditions)
    count = 0
    for maze_width, maze_height, prey_arm, maze_walls, features in conditions:
        count += 1

        # Serialize condition
        condition_string = [
            maze_width, maze_height, prey_arm, maze_walls, features]
        condition_string = _serialize(condition_string)

        # Write to file
        filename_tail = str(count).zfill(_FILENAME_ZFILL)
        filename = os.path.join(log_dir, filename_tail)
        if count % log_every == 0:
            logging.info(f'Logging maze {count} of {num_conditions}.')
        with open(filename, 'w') as f:
            json.dump(condition_string, f)
