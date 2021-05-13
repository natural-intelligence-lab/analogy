"""Eye position-based OOG task for MWorks.

The main class in this file is TaskManager, which is a task to be run by
moog.mwel.
"""

################################################################################
####  Determine whether on laptop or on rig and update sys.path accordingly
################################################################################

_PYTHON_SITE_PACKAGES = getvar('python_site_packages')
_PWD = getvar('pwd')

import sys

if '' not in sys.path:
    sys.path.insert(0, '')
if _PYTHON_SITE_PACKAGES not in sys.path:
    sys.path.append(_PYTHON_SITE_PACKAGES)
if _PWD not in sys.path:
    sys.path.append(_PWD)

################################################################################
####  Imports
################################################################################

from datetime import datetime
import importlib
import numpy as np
import os
import threading

from utils import logger_env_wrapper

from moog import environment
from moog import observers

from configs import config as config_lib


class TaskManager:
    """OOG task manager.
    
    This works for all OOG tasks in which the action space is eye position.
    """

    def __init__(self,
                 level,
                 **config_kwargs):
        """Constructor."""
        self.lock = threading.Lock()
        
        # Get config
        level_split = level.split('.')
        config_module = __import__(
            '.'.join(['configs', 'levels'] + level_split[:-1]),
            globals(),
            locals(),
            [level_split[-2]],
        )
        # Force MWorks server to reload the config, so changes to the config
        # will be propagated to MWorks without restarting the server.
        importlib.reload(config_module)
        importlib.reload(config_lib)
        # importlib.reload(set_pwd)

        config_class = getattr(config_module, level_split[-1])(
            **config_kwargs,
        )
        config = config_class()   

        # Override renderer
        image_size = (getvar('image_pixel_width'), getvar('image_pixel_height'))
        renderer = observers.PILRenderer(
            image_size=image_size,
            anti_aliasing=1,
            color_to_rgb=config['observers']['image'].color_to_rgb,
        )
        config['observers'] = {'image': renderer}

        # Create environment
        log_dir = os.path.join(_PWD, 'logs')
        self.env = logger_env_wrapper.MazeSetGoLoggingEnvironment(
            environment=environment.Environment(**config),
            log_dir=log_dir,
            metadata=[('level', level)]
        )

        # Fetch linear regression coefficients to map raw eye position to frame
        # position.
        self._eye_to_frame_coeffs = np.array(getvar('eye_to_frame_coeffs'))
        self._eye_to_frame_intercept = np.array(
            getvar('eye_to_frame_intercept'))

        self._dump_file = os.path.join(_PWD, 'dump.txt')
        with open(self._dump_file, 'w') as f:
            f.write('')

        self._end_task = False

    def reset(self):
        """Reset environment.

        This should be called at the beginning of every trial.
        """

        timestep = self.env.reset()
        if timestep is None:
            setvar('end_task', True)
            self._end_task = True

        unregister_event_callbacks()
        self.events = {}
        for varname in ('eye_x', 'eye_y'):
            self._register_event_callback(varname)

        self._keys_pressed = np.zeros(4, dtype=int)
        for varname in ('up_pressed', 'down_pressed', 'left_pressed',
                        'right_pressed'):
            self._register_event_callback(varname)

        self._space_keys_pressed = np.zeros(1, dtype=int)
        self._register_event_callback('space_pressed')
        
        self.complete = False

        # controlling image_size_x image_size_y after extracting from self.env
        image_size = self.env.meta_state['image_size']
        setvar('image_size_x', image_size)
        setvar('image_size_y', image_size)

    def _register_event_callback(self, varname):
        self.events[varname] = []
        def cb(evt):
            with self.lock:
                self.events[varname].append(evt.data)
        register_event_callback(varname, cb)

    def _get_paired_events(self, varname1, varname2):
        with self.lock:
            evt1 = self.events[varname1]
            evt2 = self.events[varname2]
            if not (evt1 and evt2):
                # No logged events.  Use current values.
                events = [[getvar(varname1), getvar(varname2)]]
            else:
                events = [[x, y] for x, y in zip(evt1, evt2)]
                # Removed "used" events, leaving any leftovers for the next pass
                evt1[:len(events)] = []
                evt2[:len(events)] = []
            return np.array(events, dtype=np.float)

    def _get_eye_action(self):
        """Get eye action."""
        eye = self._get_paired_events('eye_x', 'eye_y')
        action = eye[-1]
        action = (
            np.matmul(self._eye_to_frame_coeffs, action) +
            self._eye_to_frame_intercept
        )
        
        return action

    def _get_hand_action(self):
        """Get grid action."""

        if self.env.step_count==0:
            # Don't move on the first step
            # We set x_force and y_force to zero because for some reason the
            # joystick initially gives a non-zero action, which persists unless
            # we explicitly terminate it.
            setvar('left_pressed', 0)
            setvar('right_pressed', 0)
            setvar('down_pressed', 0)
            setvar('up_pressed', 0)
            return 4

        keys_pressed = np.array([
            getvar('left_pressed'),
            getvar('right_pressed'),
            getvar('down_pressed'),
            getvar('up_pressed'),
        ])
        if sum(keys_pressed) > 1:
            keys_pressed[self._keys_pressed] = 0
        
        if sum(keys_pressed) > 1:
            random_ind = np.random.choice(np.argwhere(keys_pressed)[:, 0])
            keys_pressed = np.zeros(4, dtype=int)
            keys_pressed[random_ind] = 1
        
        self._keys_pressed = keys_pressed

        if sum(keys_pressed):
            key_ind = np.argwhere(keys_pressed)[0, 0]
        else:
            key_ind = 4
        
        return key_ind

    def _get_hand_offline_action(self):
        """Get yes/no action."""

        if self.env.step_count == 0:
            # Don't move on the first step
            # We set x_force and y_force to zero because for some reason the
            # joystick initially gives a non-zero action, which persists unless
            # we explicitly terminate it.
            setvar('space_pressed', 0)
            return 1

        keys_pressed = np.array([
            getvar('space_pressed')
        ])
        if sum(keys_pressed) > 1:
            keys_pressed[self._space_keys_pressed] = 0

        if sum(keys_pressed) > 1:  # dealing with multiple press?
            random_ind = np.random.choice(np.argwhere(keys_pressed)[:, 0])
            keys_pressed = np.zeros(1, dtype=int)
            keys_pressed[random_ind] = 1

        self._keys_pressed = keys_pressed

        if sum(keys_pressed):
            key_ind = np.argwhere(keys_pressed)[0, 0]
        else:
            key_ind = 1

        return key_ind

    def step(self):
        """Step environment."""

        if self.complete:
            # Don't step if the task is already complete.  Returning None tells
            # MWorks that the image texture doesn't need to be updated.
            return

        if self._end_task:
            return

        eye_action = self._get_eye_action()
        hand_action = self._get_hand_action()
        hand_offline_action = self._get_hand_offline_action()
        action = {'eye': eye_action, 'hand': hand_action, 'hand_offline': hand_offline_action}

        timestep = self.env.step(action)
        reward = timestep.reward
        img = timestep.observation['image']

        if reward:
            setvar('reward_duration', reward * 1000)  # ms to us

        if timestep.last():
            setvar('end_trial', True)
            self.complete = True

        # MWorks' Python image stimulus requires a contiguous buffer, so we use
        # ascontiguousarray to provide one.
        to_return = np.ascontiguousarray(img)

        return to_return
