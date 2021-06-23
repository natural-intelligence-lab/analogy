"""Common grid_chase task config."""

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
_ITI = 15  #  250 ms
_FIXATION_STEPS = 15  #  250 ms
_AGENT_Y = 0.1
_MAZE_Y = 0.15
_MAZE_WIDTH = 0.7

_MAX_REWARDING_DIST = 0.2
_EPSILON = 1e-4  # FOR REWARD FUNCTION

_IMAGE_SIZE = [16]  # [8, 16, 24]


class TrialInitialization():

    def __init__(self, stimulus_generator, prey_lead_in, static_prey=False,
                 static_agent=False):
        self._stimulus_generator = stimulus_generator
        self._prey_lead_in = prey_lead_in
        self._static_prey = static_prey
        self._static_agent = static_agent

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

        self._meta_state = {
            'fixation_duration': 0,
            'motion_steps': 0,
            'phase': '',  # fixation -> offline -> motion -> online -> reward -> ITI
            'trial_name': '',
            'stimulus_features': stimulus['features'],
            'prey_path': prey_path,
            'prey_speed': 0,
            'maze_width': maze_width,
            'maze_height': maze_height,
            'image_size': image_size,
            'prey_distance_remaining': prey_distance_remaining,
            'RT_offline': 0,
            'tp': 0,
            'ts': 0,
        }

        return state

    def create_agent(self, state):
        agent = sprite.Sprite(
            x=0.5, y=_AGENT_Y, shape='square', aspect_ratio=0.3, scale=0.05,
            c0=128, c1=32, c2=32, metadata={'response': False, 'moved': False},
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

        # How close to center joystick must be to count as joystick centering
        self._joystick_center_threshold = 0.05

        # Compute prey speed given ms_per_unit, assuming 60 fps
        self._prey_speed = 1000. / (60. * ms_per_unit)  # 0.0083 frame width / refresh
        self._prey_lead_in = 0.08

        self._trial_init = TrialInitialization(
            stimulus_generator, prey_lead_in=self._prey_lead_in,
            static_prey=static_prey, static_agent=static_agent)

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
        )

        # joystick_center_task = tasks_custom.BeginPhase('fixation')

        # offline_task = tasks_custom.OfflineReward(
        #     'offline', max_rewarding_dist=_MAX_REWARDING_DIST)
        #
        # timeout_task = tasks.Reset(
        #     condition=lambda _, meta_state: meta_state['phase'] == 'reward',
        #     steps_after_condition=15,
        # )
        self._task = tasks.CompositeTask(
            prey_task,
            # timeout_task,
            # joystick_center_task,
            # offline_task,
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

        # 2. Joystick centering phase

        appear_joystick = gr.ModifySprites(
            ['joystick_fixation', 'joystick'], _make_opaque)

        def _should_end_joystick_fixation(state):
            joystick_pos = state['joystick'][0].position
            dist_from_center = np.linalg.norm(joystick_pos - 0.5 * np.ones(2))
            return dist_from_center < self._joystick_center_threshold

        phase_joystick_center = gr.Phase(
            one_time_rules=appear_joystick,
            end_condition=_should_end_joystick_fixation,
            name='joystick_fixation',
        )

        # 3. Fixation phase

        disappear_joystick = gr.ModifySprites(
            ['joystick_fixation', 'joystick'], _make_transparent)

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
            return (meta_state['fixation_duration'] >= _FIXATION_STEPS)

        if not self._fixation_phase:
            fixation_duration = 0
        else:
            fixation_duration = np.inf

        appear_fixation = gr.ModifySprites('fixation', _make_opaque)

        phase_fixation = gr.Phase(
            one_time_rules=[appear_fixation, disappear_joystick],
            continual_rules=[increase_fixation_dur, reset_fixation_dur],
            end_condition=_should_end_fixation,
            duration=fixation_duration,
            name='fixation',
        )

        # 4. Offline phase

        # one_time_rules
        disappear_fixation = gr.ModifySprites('fixation', _make_transparent)
        disappear_screen = gr.ModifySprites('screen', _make_transparent)
        create_agent = custom_game_rules.CreateAgent(self._trial_init)

        # continual_rules
        #   change agent color if offline reward
        def _reward(state, meta_state):
            if len(state['agent']) > 0:
                agent = state['agent'][0]
                if (meta_state['phase'] == 'offline' and
                        agent.metadata['moved'] and
                        np.all(state['agent'][0].velocity == 0)):
                    prey = state['prey'][0]
                    prey_exit_x = meta_state['prey_path'][-1][0]
                    agent_prey_dist = np.abs(agent.x - prey_exit_x)
                    reward = max(0, 1 - agent_prey_dist / (_MAX_REWARDING_DIST + _EPSILON))
                else:
                    reward = 0.
            else:
                reward = 0.
            return reward
        def _offline_reward(state, meta_state):
            return _reward(state, meta_state) > 0

        def _track_moved(s):
            if not np.all(s.velocity == 0):
                s.metadata['moved'] = True
        update_agent_metadata = gr.ModifySprites('agent', _track_moved)
        update_agent_color = gr.ConditionalRule(
            condition=lambda state, x: _offline_reward(state, x) > 0,
            rules=gr.ModifySprites('agent', _make_green)
        )
        def _increase_RT_offline(state, meta_state):
            agent = state['agent'][0]
            if not agent.metadata['moved']:
                meta_state['RT_offline'] += 1

        # end_condition
        def _end_offline_phase(state):
            agent = state['agent'][0]
            return (agent.metadata['moved'] and np.all(agent.velocity == 0))

        phase_offline = gr.Phase(
            one_time_rules=[disappear_fixation, disappear_screen, create_agent],
            continual_rules=[update_agent_metadata, update_agent_color, _increase_RT_offline],
            name='offline',
            end_condition=_end_offline_phase,  # duration=10,
        )

        # 5. Visible motion phase

        def _unglue(meta_state):
            self._maze_walk.speed = self._prey_speed
            meta_state['prey_speed'] = self._prey_speed

        unglue = gr.ModifyMetaState(_unglue)

        def _update_motion_steps(meta_state):
            meta_state['motion_steps'] += 1
            meta_state['prey_distance_remaining'] -= self._prey_speed

        update_motion_steps = gr.ModifyMetaState(_update_motion_steps)

        phase_motion_visible = gr.Phase(
            one_time_rules=unglue,
            continual_rules=update_motion_steps,
            duration=10,
            name='motion_visible',
        )

        # 6. Invisible motion phase

        def _end_motion_phase(state):
            return state['agent'][0].metadata['response']

        hide_prey = gr.ModifySprites('prey', _make_prey_transparent)

        def _increase_tp(state, meta_state):
            meta_state['tp'] += 1

        def update_ts(state, meta_state):
            meta_state['ts'] = meta_state['prey_distance_remaining'] / self._prey_speed

        phase_motion_invisible = gr.Phase(
            one_time_rules=[hide_prey, update_ts],
            continual_rules=[update_motion_steps, _increase_tp],
            end_condition=_end_motion_phase,
            name='motion_invisible',
        )

        # 7. Reward Phase

        reveal_prey = gr.ModifySprites('prey', _make_opaque)

        phase_reward = gr.Phase(
            one_time_rules=reveal_prey,
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
