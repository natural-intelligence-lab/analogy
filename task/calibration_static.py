"""Eye tracker calibration task for MWorks.

The main class in this file is CalibrationManager, which is a task to be run by
moog.mwel that fits a linear regression between the raw eye position and the
corresponding frame position.
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
from sklearn import linear_model
import threading
import time

from moog import env_wrappers
from moog import environment
from moog import observers

_MAX_TRIES_PER_POSITION = 10


def _fit_predict(train_data, test_data):
    """Fit and predit.
    
    Args:
        train_data: Numpy array of shape [num_train_trials, 2, 2].
        test_data: Numpy array of shape [num_test_trials, 2, 2].
    
    Returns:
        pred_error: Numpy array of shape [num_test_trials] containing the norm
            of the prediction error for each test trial.
        intercept: Numpy array of shape [2].
        coeffs: Numpy array of shape [2, 2].
    """
    train_eye_data = train_data[:, 0]
    train_position_data = train_data[:, 1]
    test_eye_data = test_data[:, 0]
    test_position_data = test_data[:, 1]

    reg = linear_model.LinearRegression().fit(
        train_eye_data, train_position_data)
    pred_position_data = reg.predict(test_eye_data)
    pred_error = np.linalg.norm(
        pred_position_data - test_position_data, axis=1)

    return pred_error, reg.coef_, reg.intercept_


class CalibrationManager:
    """Calibration task manager."""

    def __init__(self,
                 moog_config='static',
                 grid_size=3,
                 threshold=0.2,
                 border_width=0.15):
        """Constructor.

        Args:
            moog_config: String. Name of config file in moog_configs/ to run.
            grid_size: Int. Height and width of the grid of points.
            threshold: Float < 1. Threshold in units of frame width. If
                difference between eye-predicted and true target position is
                ever above this threshold, calibration is deemed to have failed.
            border_width: Float. How close should the perimeter points be to the
                screen borders.
        """
        self._threshold = threshold
        self.lock = threading.Lock()
        self._space_was_pressed = False

        ax_positions = np.linspace(border_width, 1. - border_width, grid_size)
        stim_positions = [(x, y) for x in ax_positions for y in ax_positions]
        self._stim_positions = [
            stim_positions[i]
            for i in np.random.permutation(len(stim_positions))
        ]
        self._eye_positions = {k: None for k in self._stim_positions}
        self._num_attempts = {k: 0 for k in self._stim_positions}
        self._stim_queue = [x for x in self._stim_positions]

        config = importlib.import_module('calibration_configs.' + moog_config)
        # Force MWorks server to reload the config, so changes to the config
        # will be propagated to MWorks without restarting the server.
        importlib.reload(config)
        config = config.get_config(None)
        image_size = (getvar('image_pixel_width'), getvar('image_pixel_height'))
        renderer = observers.PILRenderer(
            image_size=image_size,
            anti_aliasing=1,
            color_to_rgb=config['observers']['image'].color_to_rgb,
        )
        config['observers'] = {'image': renderer}
        log_dir = os.path.join(
            _PWD, 'logs/' + datetime.now().strftime('%Y_%m_%d'),
            '/'.join(moog_config.split('.')))
        self.env = env_wrappers.LoggingEnvironment(
            environment=environment.Environment(**config),
            log_dir=log_dir,
        )
        self.env.reset()

        self._dump_file = _PWD + '/dump.txt'
        with open(self._dump_file, "w") as f:
            f.write('')

    def reset(self):
        """Reset environment.

        This should be called at the beginning of every trial.
        """
        self.inter_trial_interval = False

        if len(self._stim_queue) > 0:
            self._current_position = self._stim_queue.pop()
        for s in self.env.state['agent']:
            s.opacity = 255
        self.env.step(np.array(self._current_position))
        self._num_attempts[self._current_position] += 1
        if max(self._num_attempts.values()) > _MAX_TRIES_PER_POSITION:
            setvar('task_error', True)

    def step(self):

        if self.inter_trial_interval:
            # Don't step if the task is in inter_trial_interval.  Returning None
            # tells MWorks that the image texture doesn't need to be updated.
            return

        space_pressed = getvar('space_pressed')

        if space_pressed:
            if self._space_was_pressed:
                return
            else:
                self._space_was_pressed = True

            eye_x = getvar('eye_x')
            eye_y = getvar('eye_y')
            eye = np.array([eye_x, eye_y], dtype=float)
            self._eye_positions[self._current_position] = eye
            for s in self.env.state['agent']:
                s.opacity = 0
            setvar('reward_duration', 1000)  # nanoseconds
            self._evaluate_calibration()
            self.reset()
        else:
            self._space_was_pressed = False

        img = self.env.observation()['image']
        return np.ascontiguousarray(img)

    def _evaluate_calibration(self):
        if len(self._stim_queue) > 0:
            return
        
        input_data = np.stack(
            [self._eye_positions[x] for x in self._stim_positions], axis=0)
        output_data = np.concatenate([self._stim_positions], axis=0)
        
        with open(self._dump_file, "a") as f:
            f.write(str(self._stim_positions))
            f.write(str(input_data))
            f.write(str(output_data))

        data = np.stack([input_data, output_data], axis=1)
        pred_error, coeffs, intercept = _fit_predict(data, data)
        if np.max(pred_error) < self._threshold:
            # Finished
            setvar('eye_to_frame_coeffs', coeffs.tolist())
            setvar('eye_to_frame_intercept', intercept.tolist())
            setvar('end_trial', True)
            setvar('end_task', True)
        else:
            worst_ind = np.argmax(pred_error)
            self._stim_queue = [self._stim_positions[worst_ind]]
