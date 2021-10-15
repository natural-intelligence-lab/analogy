"""Common grid_chase task config.

2021/10/10: test cue-combination
- with FP(500+exp(500)) vs no FP
- Block design (num_trials_block; 100): ABABA...
- 20 grid, # turns:[0 2 4 6], for 0 turn, [10 15 20] grids
- no joystick, just keyboard
- initial eye position for offline: present ball first
 TO DO
- orthogonalize # turns & path length?
- repeat the same trials?

"""

import abc
import collections
import math
import numpy as np

from moog import action_spaces
from moog import observers
from moog import game_rules as gr
from moog import physics as physics_lib
from moog import shapes
from moog import sprite
from moog import tasks

import maze_lib

from configs.utils import action_spaces as action_spaces_custom
from configs.utils import game_rules as custom_game_rules
from configs.utils import tasks as tasks_custom
from configs.utils import tasks_offline as tasks_custom_offline
from maze_lib.constants import max_reward, bonus_reward, reward_window

_FIXATION_THRESHOLD = 0.4
_ITI = 6  #  100 ms
_FIXATION_STEPS = 12 # 6  #  100 ms
_REWARD = 6 # 100 ms
_AGENT_Y = 0.05
_MAZE_Y = 0.1
_MAZE_WIDTH = 0.7  # 0.55

_MAX_REWARDING_DIST = 0.2
_EPSILON = 1e-4  # FOR REWARD FUNCTION

_IMAGE_SIZE = [20] # [16]  # [8, 16, 24]
_WALL_WIDTH = 0.05 # WAS 0.05 for 12 bin maze

_NUM_TRIAL_BLOCK = 5 # 100

_REFRESH_RATE = 60
_MEAN_EXP = 500/_REFRESH_RATE # 8.3
_MIN_EXP = 1000/_REFRESH_RATE  # 500/_REFRESH_RATE
_MAX_EXP = _MEAN_EXP*2


class ITrial():

    def __init__(self,init_value = 1):
        self._i_trial = init_value

    def step(self):
        self._i_trial = self._i_trial + 1

    @property
    def i_trial(self):
        return self._i_trial

class TrialInitialization():

    def __init__(self, stimulus_generator, prey_lead_in, static_prey=False,
                 static_agent=False,prey_opacity_staircase=None,i_trial_=None):
        self._stimulus_generator = stimulus_generator
        self._prey_lead_in = prey_lead_in
        self._static_prey = static_prey
        self._static_agent = static_agent
        self._prey_opacity_staircase = prey_opacity_staircase
        self._i_trial_ = i_trial_
        self._prey_factors = dict(
            shape='circle', scale=0.015, c0=0, c1=255, c2=0)
        self._fixation_shape = 0.2 * np.array([
            [-5, 1], [-1, 1], [-1, 5], [1, 5], [1, 1], [5, 1], [5, -1], [1, -1],
            [1, -5], [-1, -5], [-1, -1], [-5, -1]
        ])

    def __call__(self):
        """State initializer."""
        stimulus = self._stimulus_generator()
        if stimulus is None:
            return None

        maze_width = stimulus['maze_width']
        maze_height = stimulus['maze_height']
        prey_path = stimulus['prey_path']
        maze_walls = stimulus['maze_walls']
        num_turns = stimulus['features']['num_turns']
        maze = maze_lib.Maze(maze_width, maze_height, all_walls=maze_walls)
        cell_size = _MAZE_WIDTH / maze_width
        tunnels = maze.to_sprites(
            wall_width=_WALL_WIDTH, cell_size=cell_size, bottom_border=_MAZE_Y, c0=128,
            c1=128, c2=128)

        # Compute scaled and translated prey path
        prey_path = 0.5 + np.array(stimulus['prey_path'])
        cell_size = _MAZE_WIDTH / maze_width
        prey_path *= cell_size
        total_width = cell_size * maze_width
        prey_path += np.array([[0.5 * (1 - total_width), _MAZE_Y]])

        prey = sprite.Sprite(**self._prey_factors)

        if self._static_prey:
            prey.position = [prey_path[0][0], _AGENT_Y - 0.001]

        # Fixation cross and screen
        fixation = sprite.Sprite(
            x=0.5, y=0.5, shape=self._fixation_shape, scale=0.05,
            c0=255, c1=255, c2=255, opacity=0)
        fixation.position = [prey_path[0][0], prey_path[0][1]]

        screen = sprite.Sprite(
            x=0.5, y=0.5, shape='square', scale=2., c0=0, c1=0, c2=0)

        # Invisible eye sprite
        eye = sprite.Sprite(c0=0, c1=0, c2=0, opacity=0)

        # Joystick sprite
        joystick = sprite.Sprite(
            x=0.5, y=0.5, shape='square', aspect_ratio=0.3, scale=0.05,
            c0=32, c1=128, c2=32,
        )

        # Joystick fixation sprite
        joystick_fixation = sprite.Sprite(
            x=0.5, y=0.5, shape='circle', scale=0.015, c0=0, c1=255, c2=0,
        )

        state = collections.OrderedDict([
            ('agent', []),
            ('prey', [prey]),
            ('maze', tunnels),
            ('screen', [screen]),
            # ('joystick_fixation', [joystick_fixation]),
            # ('joystick', [joystick]),
            ('fixation', [fixation]),
            ('eye', [eye]),
        ])

        # Prey distance remaining is how far prey has to go to reach agent
        # It will be continually updated in the meta_state as the prey moves
        prey_distance_remaining = (
                self._prey_lead_in + cell_size * len(prey_path) + _MAZE_Y -
                _AGENT_Y)

        # randomly choose image size across trials
        image_size = np.random.choice(_IMAGE_SIZE)

        if self._prey_opacity_staircase is None:
            self._prey_opacity = 255
        else:
            self._prey_opacity = self._prey_opacity_staircase.opacity

        if self._i_trial_ is None:
            self._i_trial = 1
        else:
            self._i_trial = self._i_trial_._i_trial

        self._meta_state = {
            'fixation_duration': 0,
            'motion_steps': 0,
            'phase': '',  # fixation -> offline -> motion -> online -> reward -> ITI
            'trial_name': '',
            'stimulus_features': stimulus['features'],
            'prey_path': prey_path,
            'prey_speed': 0,
            'prey_opacity': self._prey_opacity,
            'maze_width': maze_width,
            'maze_height': maze_height,
            'image_size': image_size,
            'prey_distance_remaining': prey_distance_remaining,
            'prey_distance_invisible': cell_size * len(prey_path) + _MAZE_Y - _AGENT_Y,
            'slope_opacity': 0,
            'end_x_agent': 0,
            'distractor_path': None,
            'RT_offline': 0,
            'tp': 0,
            'ts': 0,
            'num_turns': num_turns,
            'id_block': bool(True),  # true/1 for odd (with FP), false/0 for even (no FP)
            'num_trial_block': _NUM_TRIAL_BLOCK,
            'i_trial': self._i_trial,
            't_offline' : 0,
        }

        return state

    def create_agent(self, state):
        agent = sprite.Sprite(
            x=self._meta_state['prey_path'][-1][0], y=_AGENT_Y, shape='square', aspect_ratio=0.3, scale=0.05, opacity = 0,
            c0=128, c1=32, c2=32, metadata={'response_up': False, 'moved_h': False,'y_speed':0},
        )
        if self._static_agent:
            agent.mass = np.inf

        state['agent'] = [agent]

    def meta_state_initializer(self):
        """Meta-state initializer."""
        return self._meta_state


class Config():
    """Callable class returning config.

    All grid chase configs should inherit from this class.
    """

    def __init__(self,
                 stimulus_generator,
                 fixation_phase=True,
                 prey_opacity=0,
                 i_trial=1,
                 i_trial_=None,
                 static_prey=False,
                 static_agent=False,
                 ms_per_unit=2000):
        """Constructor.

        Args:
            stimulus_generator: Callable returning dict with the following keys:
                * maze_arms
                * prey_arm
                * features
            fixation_phase: Bool. Whether to have a fixation phase.
            ms_per_unit: Scalar. Speed of prey. Units are milliseconds per frame
                width.
        """
        self._stimulus_generator = stimulus_generator
        self._fixation_phase = fixation_phase
        self._prey_opacity = prey_opacity
        self._static_prey = static_prey
        self._static_agent = static_agent
        self._i_trial_ = i_trial_

        if self._i_trial_ is not None:
            self._i_trial = self._i_trial_._i_trial
        else:
            self._i_trial = i_trial

        # How close to center joystick must be to count as joystick centering
        self._joystick_center_threshold = 0.05

        # Compute prey speed given ms_per_unit, assuming 60 fps
        self._prey_speed = 1000. / (60. * ms_per_unit)  # 0.0083 frame width / refresh
        self._prey_lead_in = 0.08

        self._trial_init = TrialInitialization(
            stimulus_generator, prey_lead_in=self._prey_lead_in,
            static_prey=static_prey, static_agent=static_agent,
            i_trial_=self._i_trial_)

        # Create renderer
        self._observer = observers.PILRenderer(
            image_size=(256, 256),
            anti_aliasing=1,
        )

        self._construct_action_space()
        self._construct_game_rules()
        self._construct_physics()
        self._construct_task()

    def _construct_physics(self):
        """Construct physics."""
        self._maze_walk = maze_lib.MazeWalk(
            speed=0., avatar_layer='prey', start_lead_in=self._prey_lead_in)

        if self._static_prey:
            corrective_physics = []
        else:
            corrective_physics = [self._maze_walk]

        self._physics = physics_lib.Physics(
            corrective_physics=corrective_physics)

    def _construct_task(self):
        """Construct task."""

        prey_task = tasks_custom.TimeErrorReward(
            half_width=40,  # given 60 Hz, 666*2/2 ms
            maximum=1,
            prey_speed=self._prey_speed,
            max_rewarding_dist=_MAX_REWARDING_DIST,
            prey_opacity_staircase=None,
        )

        # joystick_center_task = tasks_custom.BeginPhase('fixation')

        # offline_task = tasks_custom.OfflineReward(
        #     'offline', max_rewarding_dist=_MAX_REWARDING_DIST)
        #
        timeout_task = tasks.Reset(
            condition=lambda _, meta_state: meta_state['phase'] == 'reward',
            steps_after_condition=_REWARD,
        )
        self._task = tasks.CompositeTask(
            prey_task,
            timeout_task,
            # joystick_center_task,
            # offline_task,
        )

    def _construct_action_space(self):
        """Construct action space."""
        self._action_space = action_spaces.Composite(
            eye=action_spaces.SetPosition(action_layers=('eye',), inertia=0.),
            hand=action_spaces_custom.JoystickColor(
                up_color=(32, 128, 32), # green
                scaling_factor=0.01),
        )

    def _construct_game_rules(self):
        """Construct game rules."""

        def _set_i_trial(s):
            s.opacity = self._prey_opacity_staircase.opacity # self._prey_opacity

        def _make_transparent(s):
            s.opacity = 0

        def _make_prey_transparent(s):
            s.opacity = self._prey_opacity

        def _make_opaque(s):
            s.opacity = 255

        def _make_green(s):
            s.c0 = 32
            s.c1 = 128
            s.c2 = 32

        # 1. ITI phase

        def _reset_physics(meta_state):
            self._maze_walk.set_prey_path(meta_state['prey_path'])
            self._maze_walk.speed = 0

        reset_physics = gr.ModifyMetaState(_reset_physics)

        phase_iti = gr.Phase(
            one_time_rules=reset_physics,
            duration=_ITI,
            name='iti',
        )

        # 2. Fixation phase

        # one time (for offline phase)
        def _sample_foreperiod(meta_state):
            if meta_state['id_block']:  # true if odd (with FP)
                offline_duration = np.round(_MIN_EXP + np.min([_MAX_EXP, np.random.exponential(scale=_MEAN_EXP)]))
            else:
                offline_duration = 0
            meta_state['RT_offline'] = offline_duration
        sample_foreperiod = gr.ModifyMetaState(_sample_foreperiod)

        def _should_increase_fixation_dur(state, meta_state):
            dist = np.linalg.norm(
                state['fixation'][0].position - state['eye'][0].position)
            eye_fixating = dist < _FIXATION_THRESHOLD
            return eye_fixating

        def _increase_fixation_dur(meta_state):
            meta_state['fixation_duration'] += 1

        increase_fixation_dur = gr.ConditionalRule(
            condition=_should_increase_fixation_dur,
            rules=gr.ModifyMetaState(_increase_fixation_dur)
        )
        reset_fixation_dur = gr.ConditionalRule(
            condition=lambda state, x: not _should_increase_fixation_dur(state, x),
            rules=gr.UpdateMetaStateValue('fixation_duration', 0)
        )

        def _should_end_fixation(state, meta_state):
            return (meta_state['fixation_duration'] >= _FIXATION_STEPS)  # 6 frames 100ms

        if not self._fixation_phase:  # protocol 'random_20'
            fixation_duration = _FIXATION_STEPS # 0
        else:
            fixation_duration = np.inf

        appear_fixation = gr.ModifySprites('fixation', _make_opaque)

        phase_fixation = gr.Phase(
            one_time_rules=[appear_fixation,sample_foreperiod],
            continual_rules=[increase_fixation_dur, reset_fixation_dur],
            end_condition=_should_end_fixation,
            duration=fixation_duration,
            name='fixation',
        )

        # 3. Offline phase

        # one_time_rules
        disappear_fixation = gr.ModifySprites('fixation', _make_transparent)
        disappear_screen = gr.ModifySprites('screen', _make_transparent)
        create_agent = custom_game_rules.CreateAgent(self._trial_init)

        def _increase_t_offline(meta_state):
            meta_state['t_offline'] += 1
        increase_t_offline = gr.ModifyMetaState(_increase_t_offline)

        def _should_end_offline(state, meta_state):
            return (meta_state['t_offline'] >= meta_state['RT_offline'])  # 6 frames 100ms

        phase_offline = gr.Phase(
            one_time_rules=[disappear_fixation, disappear_screen, create_agent],
            name='offline',
            continual_rules=increase_t_offline,
            end_condition=_should_end_offline,
        )

        # 4. Visible motion phase

        def _unglue(meta_state):
            self._maze_walk.speed = self._prey_speed
            meta_state['prey_speed'] = self._prey_speed
        unglue = gr.ModifyMetaState(_unglue)

        glue_agent = custom_game_rules.GlueAgent()

        def _update_motion_steps(meta_state):
            meta_state['motion_steps'] += 1
            meta_state['prey_distance_remaining'] -= self._prey_speed

        update_motion_steps = gr.ModifyMetaState(_update_motion_steps)

        def _end_vis_motion_phase(state,meta_state):
            if meta_state['motion_steps'] > (self._prey_lead_in / self._prey_speed):
                return True
            return False

        phase_motion_visible = gr.Phase(
            one_time_rules=[unglue,glue_agent],
            continual_rules=update_motion_steps,
            end_condition=_end_vis_motion_phase,  #  duration=10,
            name='motion_visible',
        )

        # 5. Invisible motion phase

        def _end_motion_phase(state):
            return state['agent'][0].metadata['response_up']

        hide_prey = gr.ModifySprites('prey', _make_prey_transparent)

        def _increase_tp(meta_state):
            meta_state['tp'] += 1
        increase_tp = gr.ModifyMetaState(_increase_tp)

        def _update_ts(meta_state):
            meta_state['ts'] = meta_state['prey_distance_remaining'] / self._prey_speed
        update_ts = gr.ModifyMetaState(_update_ts)

        phase_motion_invisible = gr.Phase(
            one_time_rules=[hide_prey, update_ts],
            continual_rules=[update_motion_steps, increase_tp],
            end_condition=_end_motion_phase,
            name='motion_invisible',
        )

        # 6. Reward Phase

        reveal_prey = gr.ModifySprites('prey', _make_opaque)
        opaque_agent = gr.ModifySprites('agent', _make_opaque)

        def _set_id_block(meta_state):
            if self._i_trial_ is not None:
                self._i_trial_.step()
            meta_state['i_trial'] = self._i_trial
            i_block = np.floor_divide(meta_state['i_trial'],_NUM_TRIAL_BLOCK) # 1234...
            meta_state['id_block'] = bool((i_block % 2) == 0) # true for odd (with FP), false for even
        set_id_block = gr.ModifyMetaState(_set_id_block)

        phase_reward = gr.Phase(
            one_time_rules=[set_id_block,reveal_prey,opaque_agent],
            continual_rules=[update_motion_steps],
            name='reward',
        )

        # Final rules
        phase_sequence = gr.PhaseSequence(
            phase_iti,
            phase_fixation,
            phase_offline,
            phase_motion_visible,
            phase_motion_invisible,
            phase_reward,
            meta_state_phase_name_key='phase',
        )
        self._game_rules = (phase_sequence,)

    def __call__(self):
        """Return config."""

        config = {
            'state_initializer': self._trial_init,
            'physics': self._physics,
            'task': self._task,
            'action_space': self._action_space,
            'observers': {'image': self._observer},
            'game_rules': self._game_rules,
            'meta_state_initializer': self._trial_init.meta_state_initializer,
        }
        return config
