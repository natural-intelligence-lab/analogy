"""Common grid_chase task config.

TO DO:
1) for offline+online (H), no online if no offline reward
2) how to change _MAX_REWARDING_DIST during task running? _MAX_REWARDING_DIST
3) change initial agent position? configs/levels/training/vertical_timing.py

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
_ITI = 60
_FIXATION_STEPS = 60  # 30
_AGENT_Y = 0.1
_MAZE_Y = 0.15
_MAZE_WIDTH = 0.7

_MAX_REWARDING_DIST=0.15
_EPSILON=1e-4 # FOR REWARD FUNCTION

_MAX_WAIT_TIME_GAIN = 2 # when tp>2*ts, abort
_JOYSTICK_FIXATION_POSTOFFLINE = 36 # 600

_IMAGE_SIZE = [24]  # [8, 16, 24]

# _STEP_OPACITY = 40  # [0 255]
_STEP_OPACITY_UP = 2 # 3 # 10  # [0 255]
_STEP_OPACITY_DOWN = 5 # 30 # 40  # [0 255]

_REWARD = 6 # 100 ms # post zero prey_distance
_TOOTH_HALF_WIDTH = 40

class PreyOpacityStaircase():

    def __init__(self,
                 init_value=10,
                 success_delta=_STEP_OPACITY_DOWN,
                 failure_delta=_STEP_OPACITY_UP,
                 minval=0,
                 maxval=255):
        self._opacity = init_value
        self._success_delta = success_delta
        self._failure_delta = failure_delta
        self._minval = minval
        self._maxval = maxval

    def step(self, reward):
        if reward > 0:
            self._opacity = max(self._opacity - self._success_delta, self._minval)
        elif reward <= 0:
            self._opacity = min(self._opacity + self._failure_delta, self._maxval)

    @property
    def opacity(self):
        return self._opacity

class TrialInitialization():

    def __init__(self, stimulus_generator, prey_lead_in, prey_speed, static_prey=False,
                 static_agent=False,prey_opacity_staircase=None):
        self._stimulus_generator = stimulus_generator
        self._prey_lead_in = prey_lead_in
        self._prey_speed = prey_speed
        self._static_prey = static_prey
        self._static_agent = static_agent
        self._prey_opacity_staircase=prey_opacity_staircase
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
        maze = maze_lib.Maze(maze_width, maze_height, all_walls=maze_walls)
        cell_size = _MAZE_WIDTH / maze_width
        tunnels = maze.to_sprites(
            wall_width=0.05, cell_size=cell_size, bottom_border=_MAZE_Y, c0=128,
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
            ('joystick_fixation', [joystick_fixation]),
            ('joystick', [joystick]),
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

        self._meta_state = {
            'fixation_duration': 0,
            'motion_steps': 0,
            'phase': '',  # fixation -> offline -> motion -> online -> reward -> ITI
            'trial_name': '',
            'stimulus_features': stimulus['features'],
            'prey_path': prey_path,
            'prey_speed': self._prey_speed,
            'prey_opacity': self._prey_opacity,
            'half_width' : _TOOTH_HALF_WIDTH,
            'maze_width': maze_width,
            'maze_height': maze_height,
            'image_size': image_size,
            'prey_distance_remaining': prey_distance_remaining,
            'RT_offline': 0,
            'tp': 0,
            'ts': 0,
            'max_rewarding_dist': _MAX_REWARDING_DIST,
            'joystick_fixation_postoffline': 0
        }

        return state

    def create_agent(self, state):
        agent = sprite.Sprite(
            x=0.5, y=_AGENT_Y, shape='square', aspect_ratio=0.3, scale=0.05,
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
                 prey_opacity_staircase=None,
                 fixation_phase=True,
                 prey_opacity=0,
                 static_prey=False,
                 static_agent=False,
                 ms_per_unit=2000,
                 ):
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
        self._static_prey = static_prey
        self._static_agent = static_agent
        self._prey_opacity_staircase = prey_opacity_staircase

        if self._prey_opacity_staircase is not None:
            self._prey_opacity = self._prey_opacity_staircase.opacity
        else:
            self._prey_opacity = prey_opacity

        # How close to center joystick must be to count as joystick centering
        self._joystick_center_threshold = 0.05

        # Compute prey speed given ms_per_unit, assuming 60 fps
        self._prey_speed = 1000. / (60. * ms_per_unit) # 0.0083 frame width / refresh
        self._prey_lead_in = 0.15  # 0.08

        self._trial_init = TrialInitialization(
            stimulus_generator, prey_lead_in=self._prey_lead_in, prey_speed=self._prey_speed,
            static_prey=static_prey, static_agent=static_agent, prey_opacity_staircase=self._prey_opacity_staircase)

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
             half_width=_TOOTH_HALF_WIDTH, # 40,  # given 60 Hz, 666*2/2 ms
             maximum=1,
             prey_speed=self._prey_speed,
             max_rewarding_dist = _MAX_REWARDING_DIST,
             prey_opacity_staircase = self._prey_opacity_staircase,
        )

        # joystick_center_task = tasks_custom.BeginPhase('fixation')

        offline_task = tasks_custom.OfflineReward(
            'offline', max_rewarding_dist=_MAX_REWARDING_DIST)  # 0.1

        timeout_task = tasks.Reset(
            condition=lambda _, meta_state: meta_state['phase'] == 'reward' 
            and meta_state['prey_distance_remaining']<0, # to prevent abort for H
            steps_after_condition=_REWARD,
        )
        self._task = tasks.CompositeTask(
            # joystick_center_task,
            offline_task,
            timeout_task,
            prey_task,
        )

    def _construct_action_space(self):
        """Construct action space."""
        self._action_space = action_spaces.Composite(
            eye=action_spaces.SetPosition(action_layers=('eye',), inertia=0.),
            hand=action_spaces_custom.JoystickColor(
                up_color=(128, 32, 32),  # red # (32, 128, 32), # green
                scaling_factor=0.01),
        )

    def _construct_game_rules(self):
        """Construct game rules."""

        def _make_transparent(s):
            s.opacity = 0

        def _set_prey_opacity(s):
            s.opacity = self._prey_opacity_staircase.opacity # self._prey_opacity

        def _make_opaque(s):
            s.opacity=255

        def _make_green(s):
            s.c0 = 32
            s.c1 = 128
            s.c2 = 32

        def _make_red(s):
            s.c0 = 128
            s.c1 = 32
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

        # 2. Joystick centering phase

        appear_joystick = gr.ModifySprites(
            ['joystick_fixation', 'joystick'], _make_opaque)

        def _should_end_joystick_fixation(state):
            if state is not None:
                if state['joystick'] is not None:
                    joystick_pos = state['joystick'][0].position
                    dist_from_center = np.linalg.norm(joystick_pos - 0.5 * np.ones(2))
                    return dist_from_center < self._joystick_center_threshold
            else:
                return false

        phase_joystick_center = gr.Phase(
            one_time_rules=appear_joystick,
            end_condition=_should_end_joystick_fixation,
            name='joystick_fixation',
        )

        # 3. Fixation phase

        # one_time_rules
        appear_fixation = gr.ModifySprites('fixation', _make_opaque)
        disappear_joystick = gr.ModifySprites(
            ['joystick_fixation', 'joystick'], _make_transparent)

        # continual_rules
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

        # end_condition
        def _should_end_fixation(state, meta_state):
            return (meta_state['fixation_duration'] >= _FIXATION_STEPS)

        if not self._fixation_phase:
            fixation_duration = 0
        else:
            fixation_duration = np.inf

        phase_fixation = gr.Phase(
            one_time_rules=[appear_fixation, disappear_joystick],
            continual_rules=[increase_fixation_dur, reset_fixation_dur],
            end_condition=_should_end_fixation,
            duration=fixation_duration,
            name='fixation',
        )

        # 4. Offline phase

        # one_time_rules
        create_agent = custom_game_rules.CreateAgent(self._trial_init)
        disappear_fixation = gr.ModifySprites('fixation', _make_transparent)
        disappear_screen = gr.ModifySprites('screen', _make_transparent)

        # continual_rules
        # change agent color if offline reward
        def _reward(state, meta_state):
            if len(state['agent']) > 0:
                agent = state['agent'][0]
                if (meta_state['phase'] == 'offline' and
                        agent.metadata['moved_h'] and
                        np.all(state['agent'][0].velocity == 0)): ##
                    prey_exit_x = meta_state['prey_path'][-1][0]
                    reward = max(0, 1 - np.abs(agent.x - prey_exit_x) / (_MAX_REWARDING_DIST + _EPSILON))
                else:
                    reward = 0.
            else:
                reward = 0.
            return reward
        def _offline_reward(state, meta_state):
            return _reward(state, meta_state) > 0
        update_agent_color = gr.ConditionalRule(
            condition=lambda state, x: _offline_reward(state, x)>0,
            rules=gr.ModifySprites('agent', _make_green)
        )

        def _track_moved_h(s):
            if not np.all(s.velocity[0] == 0): ##
                s.metadata['moved_h'] = True
        update_agent_metadata = gr.ModifySprites('agent', _track_moved_h)

        def _should_increase_RT_offline(state, meta_state):
            agent = state['agent'][0]
            return not agent.metadata['moved_h']
        def _increase_RT_offline(meta_state):
            meta_state['RT_offline'] += 1
        update_RT_offline = gr.ConditionalRule(
            condition=_should_increase_RT_offline,
            rules=gr.ModifyMetaState(_increase_RT_offline)
        )
        # end_condition
        # def _should_increase_joystick_fixation_dur(state,meta_state):
        #     if len(state['agent']) > 0:
        #         agent = state['agent'][0]
        #         return (meta_state['phase'] == 'offline' and agent.metadata['moved_h'] and np.all(agent.velocity == 0))
        # def _increase_joystick_fixation_dur(meta_state):
        #     meta_state['joystick_fixation_postoffline'] += 1
        # update_joystick_fixation_dur = gr.ConditionalRule(
        #     condition=_should_increase_joystick_fixation_dur,
        #     rules=gr.ModifyMetaState(_increase_joystick_fixation_dur)
        # )
        def _end_offline_phase(state,meta_state):
            agent = state['agent'][0]
            return agent.metadata['moved_h'] and np.all(agent.velocity == 0) and agent.metadata['y_speed'] == 0 ##
            # meta_state['joystick_fixation_postoffline']>_JOYSTICK_FIXATION_POSTOFFLINE # np.all(agent.velocity == 0) # 

        phase_offline = gr.Phase(
            one_time_rules=[disappear_fixation, disappear_screen, create_agent],
            continual_rules=[update_agent_metadata, update_RT_offline, update_agent_color], # ,update_joystick_fixation_dur],  # update_agent_color 
            name='offline',
            end_condition=_end_offline_phase,  #  duration=10,
        )

        # 5. Visible motion phase

        def _unglue(meta_state):
            self._maze_walk.speed = self._prey_speed
            meta_state['prey_speed'] = self._prey_speed
        unglue = gr.ModifyMetaState(_unglue)

        glue_agent = custom_game_rules.GlueAgent()
        make_agent_red = gr.ModifySprites('agent', _make_red)


        def _update_motion_steps(meta_state):
            meta_state['motion_steps'] += 1
            meta_state['prey_distance_remaining'] -= self._prey_speed
        update_motion_steps = gr.ModifyMetaState(_update_motion_steps)

        def _end_vis_motion_phase(state,meta_state):
            if meta_state['motion_steps'] > (self._prey_lead_in / self._prey_speed):
                return True
            return False

        phase_motion_visible = gr.Phase(
            one_time_rules=[unglue,glue_agent,make_agent_red],
            continual_rules=update_motion_steps,
            end_condition=_end_vis_motion_phase,  # duration=10,
            name='motion_visible',
        )

        # 6. Invisible motion phase
        set_prey_opacity = gr.ModifySprites('prey', _set_prey_opacity)  # self._prey_opacity
        def _update_ts(meta_state):
            meta_state['ts'] = meta_state['prey_distance_remaining'] / self._prey_speed # [Hz]
        update_ts = gr.ModifyMetaState(_update_ts)
        def _increase_tp(meta_state):
            meta_state['tp'] += 1
        increase_tp = gr.ModifyMetaState(_increase_tp)

        def _end_motion_phase(state,meta_state):
            id_response_up = state['agent'][0].metadata['response_up']
            id_late = meta_state['tp'] > _MAX_WAIT_TIME_GAIN*meta_state['ts']
            return id_response_up or id_late

        phase_motion_invisible = gr.Phase(
            one_time_rules=[set_prey_opacity,update_ts],
            continual_rules=[update_motion_steps,increase_tp],
            end_condition=_end_motion_phase,
            name='motion_invisible',
        )

        # 7. Reward Phase

        reveal_prey = gr.ModifySprites('prey', _make_opaque)
        make_agent_green = gr.ModifySprites('agent', _make_green)

        def _id_time_reward(state,meta_state):
            time_remaining = meta_state['prey_distance_remaining'] / meta_state['prey_speed']
            time_error = np.abs(time_remaining)
            return time_error > meta_state['half_width']
            
        update_prey_color = gr.ConditionalRule(
            condition=lambda state, x: _id_time_reward(state, x),
            rules=gr.ModifySprites('prey', _make_red)
        )

        phase_reward = gr.Phase(
            one_time_rules=[reveal_prey,make_agent_green,update_prey_color],
            continual_rules=update_motion_steps,
            name='reward',
        )

        # Final rules
        phase_sequence = gr.PhaseSequence(
            phase_iti,
            phase_joystick_center,
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
