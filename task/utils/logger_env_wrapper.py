"""Environment wrapper class for logging."""

import copy
from datetime import datetime
import json
import logging
import numpy as np
import os
import time

from moog import env_wrappers
from moog import sprite
import maze_lib

# This is the number of numerals in filenames. Since there is one file per
# episode, you should pick _FILENAME_ZFILL large enough that the number of
# episodes in your dataset is less than 10^_FILENAME_ZFILL.
_FILENAME_ZFILL = 5


class VertexLogging():
    NEVER = 'NEVER'
    ALWAYS = 'ALWAYS'
    WHEN_NECESSARY = 'WHEN_NECESSARY'


def _serialize_action(action):
    """Serialize an action.
    
    Numpy arrays are not JSON serializable, so we must convert numpy arrays to
    lists in actions. This function is recursive to handle actions that are 
    arbitrarily nested dictionaries of numpy arrays, which may arise in the case
    of multi-agent action spaces.

    Args:
        action: Action from an environment timestep.

    Returns:
        action: Serialized action that can be JSON dumped.
    """
    if isinstance(action, np.int_):
        action = int(action)
    elif isinstance(action, np.float_):
        action = float(action)
    elif isinstance(action, np.ndarray):
        action = action.tolist()
    elif isinstance(action, dict):
        action = {k: _serialize_action(v) for k, v in action.items()}
    return action


class MazePongLoggingEnvironment(env_wrappers.AbstractEnvironmentWrapper):
    """Environment class for logging timesteps.
    
    This logger produces a description of the log in 'description.txt' of 
    log_dir, so please refer to that for a detailed account of the structure of
    the logs.
    """
    
    def __init__(self, environment, log_dir='logs', metadata=None):
        """Constructor.

        Args:
            environment: Instance of moog.Environment.
            log_dir: String. Log directory relative to working directory.
            metadata: Optional metadata to log.
        """
        super(MazePongLoggingEnvironment, self).__init__(environment)

        # Set the logging directory
        now_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if log_dir[0] == '/':
            log_dir = os.path.join(log_dir, now_str)
        else:
            log_dir = os.path.join(os.getcwd(), log_dir, now_str)
        os.makedirs(log_dir)
        self._log_dir = log_dir

        # Log metadata
        with open(os.path.join(self._log_dir, 'metadata.txt'), 'w') as f:
            json.dump(metadata, f)

        # These are the attributes that we'll log at the beginning of episodes
        self._attributes_full = list(sprite.Sprite.FACTOR_NAMES)

        # Log full attribute list
        attr_full_filename = os.path.join(self._log_dir, 'attributes_full.txt')
        logging.info('Logging full attribute list {} to {}.'.format(
            self._attributes_full, attr_full_filename))
        with open(attr_full_filename, 'w') as f:
            json.dump(self._attributes_full, f)
        
        # These are the attributes that we'll log in the middle of episodes
        self._attributes_partial = [
            'x', 'y', 'x_vel', 'y_vel', 'opacity', 'metadata']

        # Log partial attribute list
        attr_partial_filename = os.path.join(
            self._log_dir, 'attributes_partial.txt')
        logging.info('Logging partial attribute list {} to {}.'.format(
            self._attributes_partial, attr_partial_filename))
        with open(attr_partial_filename, 'w') as f:
            json.dump(self._attributes_partial, f)

        # Log description and initialize self._episode_count
        self._log_description()
        self._episode_count = 0

    def _log_description(self):
        """Log a description of the data to a description.txt file."""
        description_filename = os.path.join(self._log_dir, 'description.txt')
        logging.info('Logging description to {}.'.format(description_filename))
        description = (
            'Each numerical file in this directory is an episode of the task. '
            'Each such file contains a json-serialized list.'
            '\n\n'
            'The first element of this list is a serialized full state of the '
            'environment. This is a list, each element of which represents a '
            'layer in the environment state. The layer is represented as a '
            'list [k, [], [], [], ...], where k is the layer name and the '
            'subsequent elements are serialized sprites. Each serialized '
            'sprite is a list of attributes. See attributes_full.txt for the '
            'attributes contained.'
            '\n\n'
            'The second element of the top-level list is a serialized maze, in '
            'the form of a boolean array.'
            '\n\n'
            'The subsequent elements represent steps in the episode. Each step '
            'is a list of four elements, [[`time`, time], [`reward`, reward], '
            '[`step_type`, step_type], [`action`, action], [`meta_state`, '
            'meta_state], state].'
            '\n'
            'time is a timestamp of the timestep.'
            '\n'
            'reward contains the value of the reward at that step.'
            '\n'
            'step_type indicates the dm_env.StepType of that step, i.e. '
            'whether it was first, mid, or last.'
            '\n'
            'action contains the agent action for the step.'
            '\n'
            'meta_state is the serialized meta_state of the environment.'
            '\n'
            'state is a list, each element of which represents a layer in the '
            'environment state. The layer is represented as a list [k, [], [], '
            '[], ...], where k is the layer name and the subsequent elements '
            'are partially serialized sprites. Each partially serialized '
            'sprite is a list of attributes. See attributes_partial.txt for '
            'the attributes contained.'
        )
        with open(description_filename, 'w') as f:
            f.write(description)

    def _serialize(self, x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, (np.float32, np.float64)):
            return float(x)
        elif isinstance(x, (np.int32, np.int64)):
            return int(x)
        elif isinstance(x, list):
            return [self._serialize(a) for a in x]
        elif isinstance(x, tuple):
            return (self._serialize(a) for a in x)
        elif isinstance(x, dict):
            return {k: self._serialize(v) for k, v in x.items()}
        else:
            return x

    def _serialize_sprite_full(self, s):
        """Serialize a sprite as a list of attributes."""
        attributes = [
            self._serialize(getattr(s, x)) for x in self._attributes_full]
        attributes.append(s.vertices.tolist())
        return attributes

    def _serialized_state_full(self):
        """Serialized a state."""
        serialized_state = [
            [k, [self._serialize_sprite_full(s) for s in self.state[k]]]
            for k in self.state
        ]
        return serialized_state

    def _serialize_sprite_partial(self, s):
        """Serialize a sprite as a list of attributes."""
        attributes = [
            self._serialize(getattr(s, x)) for x in self._attributes_partial]
        return attributes

    def _serialized_state_partial(self):
        """Just serialize object positions, velocities, and angles."""
        def _serialize_state(k):
            x = [k, [self._serialize_sprite_partial(s) for s in self.state[k]]]
            return x

        serialize_layers = ['agent', 'prey', 'eye', 'screen', 'fixation']
        serialized_state = [_serialize_state(k) for k in serialize_layers]

        return serialized_state

    def reset(self):
        timestep = self._environment.reset()
        if timestep is None:
            return None

        # Add serialized state to log
        serialized_state = self._serialized_state_full()
        self._episode_log = [serialized_state]

        # Add maze to log
        self._episode_log.append(self.meta_state['maze_matrix'].tolist())

        return timestep

    def step(self, action):
        """Step the environment with an action, logging timesteps."""
        timestep = self._environment.step(action)
        action = _serialize_action(action)
        str_timestep = (
            [['time', time.time()],
             ['reward', timestep.reward],
             ['step_type', timestep.step_type.value],
             ['action', action],
             ['meta_state', self._serialize(self._environment.meta_state)],
             self._serialized_state_partial()]
        )
        self._episode_log.append(str_timestep)
        
        if timestep.last():
            # Write the episode to a log file
            episode_count_str = str(self._episode_count).zfill(_FILENAME_ZFILL)
            filename = os.path.join(self._log_dir, episode_count_str)
            logging.info('Logging episode {} to {}.'.format(
                self._episode_count, filename))
            with open(filename, 'w') as f:
                json.dump(self._episode_log, f)
            self._episode_count += 1
            self._episode_log = []
            
        return timestep