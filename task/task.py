"""Eye position-based OOG task for MWorks.

The main class in this file is TaskManager, which is a task to be run by
moog.mwel.
"""

from mworkscore import getvar, setvar

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
import time

from utils import logger_env_wrapper

from moog import environment
from moog import observers

if getvar('platform') == 'laptop':
    # from configs import config as config_lib
    from configs import config_human as config_lib
elif getvar('platform') == 'desktop':
    # from configs import config_g as config_lib
    from configs import config_human as config_lib
    # from configs import config_human as config_lib
elif getvar('platform') == 'psychophysics':
    from configs import config_human as config_lib
elif getvar('platform') == 'monkey_ephys':
    from configs import config as config_lib
elif getvar('platform') == 'monkey_train':
    # from configs import config_g as config_lib
    from configs import config as config_lib
    # from configs import config_online as config_lib    
    

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

        # staircase
        if getvar('platform') == 'monkey_ephys' or getvar('platform') == 'monkey_train' or getvar('platform') == 'laptop':
            self._prey_opacity_staircase = config_class._prey_opacity_staircase
            self._path_prey_opacity_staircase = config_class._path_prey_opacity_staircase
            self._path_prey_position_staircase = config_class._path_prey_position_staircase
            self._update_p_correct = config_class._update_p_correct
            self._reward_window_staircase = config_class._reward_window_staircase

        # i_trial dynamics
        if getvar('platform') == 'laptop' or getvar('platform') == 'psychophysics' or getvar('platform') == 'desktop':
            self._id_trial_staircase = config_class._id_trial_staircase

        # Create environment
        log_dir = os.path.join(_PWD, 'logs')
        self.env = logger_env_wrapper.MazeSetGoLoggingEnvironment(
            environment=environment.Environment(**config),
            log_dir=log_dir,
            metadata=[('level', level)]
        )

        # Fetch linear regression coefficients to map raw eye position to frame
        # position.
        self._eye_to_frame_coeffs = np.array(getvar('eye_to_frame_coeffs'))  # [[1, 0], [0, 1]] in moog.mwel
        self._eye_to_frame_intercept = np.array(
            getvar('eye_to_frame_intercept'))  # [0, 0] in moog.mwel

        self._dump_file = os.path.join(_PWD, 'dump.txt')
        with open(self._dump_file, 'w') as f:
            f.write('')

        self._end_task = False

        # time stamp
        self.flag1 = True
        self.flag2 = True
        self.flag3 = True
        self.flag4 = True
        self.flag5 = True
        self.flag6 = True
        self.flag7 = True
        self.flag8 = True
        self.flag9 = True

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

        # self._keys_pressed = np.zeros(4, dtype=int)
        # for varname in ('up_pressed', 'down_pressed', 'left_pressed',
        #                 'right_pressed'):
        #     self._register_event_callback(varname)

        self._space_keys_pressed = np.zeros(1, dtype=int)
        self._register_event_callback('space_pressed')
        
        self.complete = False

        # controlling image_size_x image_size_y after extracting from self.env
        image_size = self.env.meta_state['image_size']
        setvar('image_size_x', image_size)
        setvar('image_size_y', image_size)

        # time stamp
        self.flag1 = True
        self.flag2 = True
        self.flag3 = True
        self.flag4 = True
        self.flag5 = True
        self.flag6 = True
        self.flag7 = True
        self.flag8 = True
        self.flag9 = True

        self.flag_turnpoint = 0

        if getvar('platform') == 'monkey_ephys' or getvar('platform') == 'monkey_train' or getvar('platform') == 'laptop':
            setvar('prey_opacity',self._prey_opacity_staircase.opacity)
            setvar('path_prey_opacity', self._path_prey_opacity_staircase.opacity)
            setvar('num_trial_junction', self._update_p_correct.n_trial)
            setvar('num_trial_amb_junction', self._update_p_correct.n_trial_amb)
            setvar('num_correct_junction', self._update_p_correct.n_correct_n_junction)
            setvar('num_correct_amb_junction', self._update_p_correct.n_correct_n_amb_junction)
            setvar('p_visible_aid', self._path_prey_position_staircase.path_prey_position)
            setvar('reward_window', self._reward_window_staircase._max_wait_time_gain)

        # TBD: debug - how to change _MAX_REWARDING_DIST during task running?
        # max_rewarding_dist = getvar('max_rewarding_dist')
        # self.env.meta_state['max_rewarding_dist'] = max_rewarding_dist

        # ?
        setvar('prey_opacity',self.env.meta_state['prey_opacity']) ## updated in task.reward
        setvar('path_prey_opacity', self.env.meta_state['path_prey_opacity'])  ## updated in task.reward

        # set trial & block
        # i_trial dynamics
        if getvar('platform') == 'laptop' or getvar('platform') == 'psychophysics' or getvar('platform') == 'desktop':
            setvar('num_completeTrials',self.env.meta_state['i_trial'])
            setvar('id_block', self.env.meta_state['id_block']) # true/1 for odd (with FP), false/0 for even (no FP)
            setvar('num_trials_block', self.env.meta_state['num_trial_block'])

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
        """Get joystick action."""
        if self.env.step_count==0:
            # Don't move on the first step
            # We set x_force and y_force to zero because for some reason the
            # joystick initially gives a non-zero action, which persists unless
            # we explicitly terminate it.
            setvar('x_force', 0.)
            setvar('y_force', 0.)
            return np.zeros(2)
        else:
            return np.array([getvar('x_force'), getvar('y_force')])

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
        action = {'eye': eye_action, 'hand': hand_action}

        timestep = self.env.step(action)
        reward = timestep.reward
        img = timestep.observation['image']

        if reward:
            setvar('reward_duration', reward)  # ms to us

        if timestep.last():
            setvar('end_trial', True)
            self.complete = True
            # RT_offline=self.env.meta_state['RT_offline']/60  # [s]
            # setvar('RT_offline',RT_offline)
            # tp = self.env.meta_state['tp']/60
            # setvar('tp', tp)

        # 3.phase_fixation, 'fixation' (0)
        if self.env.meta_state['phase'] == 'fixation' and self.flag1:
            setvar('tFix',time.time())
            setvar('prey_distance_invisible',self.env.meta_state['prey_distance_invisible'])
            self.flag1 = False

        # 4.phase_ball_on,  'ball_on' (.5sec)
        if self.env.meta_state['phase'] == 'ball_on' and self.flag2:
            setvar('tBallOn',time.time())
            self.flag2 = False

        # 5.phase_maze_on,  'maze_on' (0)
        if self.env.meta_state['phase'] == 'maze_on' and self.flag3:
            tMazeOn = time.time()
            setvar('tMazeOn',tMazeOn)
            self.flag3 = False

        # 6.phase_path_prey, 'path_prey' (0)
        if self.env.meta_state['phase'] == 'path_prey' and self.flag4:
            setvar('tPathPrey',time.time())
            self.flag4 = False

        # 7. phase_off_move, 'offMove' (variable)
        if self.env.meta_state['phase'] == 'offMove' and self.flag5:
            tOffline = time.time()
            setvar('tOffline',tOffline)
            self.flag5 = False
            setvar('num_turns',self.env.meta_state['num_turns'])
            setvar('end_x_prey',self.env.meta_state['prey_path'][-1][0])
            setvar('start_x_prey',self.env.meta_state['prey_path'][0][0])

            # if self.env.meta_state['distractor_path'] is not None:
            #     if not len(self.env.meta_state['distractor_path'])==0:
            #         setvar('end_x_distract',self.env.meta_state['distractor_path'][-1][0])

        # get offline RT
        agent = self.env.state['agent']
        if self.env.meta_state['phase'] == 'offMove' and self.flag6:
            if len(agent) > 0:
                if agent[0].metadata['moved_h']:
                    tOfflineRT = time.time()
                    setvar('tOfflineRT',tOfflineRT)
                    tMazeOn = getvar('tMazeOn')
                    RT_offline = tOfflineRT - tMazeOn # tOffline # tMazeOn if maze_on duration is zero, no tMazeOn
                    # RT_offline = self.env.meta_state['RT_offline'] / 60  # [s] # less accurate by skipping monitor refresh? but still constrained by 60 Hz?
                    setvar('RT_offline', RT_offline)
                    self.flag6 = False

        # 8. phase_motion_visible, 'motion_visible' (variable)
        if self.env.meta_state['phase'] == 'motion_visible' and self.flag7:
            setvar('id_correct_offline',self.env.meta_state['id_correct_offline'])
            setvar('tVisMotion',time.time())
            setvar('end_x_agent',self.env.meta_state['end_x_agent'])
            self.flag7 = False

        # 9. phase_motion_invisible, 'motion_invisible' (variable)
        if self.env.meta_state['phase'] == 'motion_invisible' and self.flag8:
            tInvMotion = time.time()
            setvar('tInvMotion',tInvMotion)
            self.flag8 = False
        if self.env.meta_state['phase'] == 'motion_invisible':
            if getvar('platform') == 'monkey_ephys' or getvar('platform') == 'monkey_train':
                setvar('slope_opacity',self.env.meta_state['slope_opacity'])
                # turnpoint
                turnpoint = self.env.state['turnpoint']
                if len(turnpoint) > 0:
                    if turnpoint[self.flag_turnpoint].metadata['contacted'] == 1:
                        setvar('id_turn', 1)
                        self.flag_turnpoint += 1
                # setvar('prey_opacity',self.env.state['prey'][0].opacity)

        # 10. phase_reward, 'reward'
        if self.env.meta_state['phase'] == 'reward' and self.flag9:
            tRew = time.time()
            setvar('tRew',tRew)
            self.flag9 = False
            tInvMotion = getvar('tInvMotion')
            setvar('tp',tRew-tInvMotion)
            ts = self.env.meta_state['ts']/60 # [s]
            setvar('ts', ts)

        # MWorks' Python image stimulus requires a contiguous buffer, so we use
        # ascontiguousarray to provide one.
        to_return = np.ascontiguousarray(img)

        return to_return

        # 1.phase_iti, 'iti' (1sec)
        # 2.phase_joystick_center, 'joystick_fixation' (variable)
        # 3.phase_fixation, 'fixation' (0)
        # 4.phase_ball_on,  'ball_on' (.5sec)
        # 5.phase_maze_on,  'maze_on' (0)
        # 6.phase_path_prey, 'path_prey' (0)
        # 7. phase_off_move, 'offMove' (variable)
        # 8. phase_motion_visible, 'motion_visible' (variable)
        # 9. phase_motion_invisible, 'motion_invisible' (variable)
        # 10. phase_reward, 'reward'