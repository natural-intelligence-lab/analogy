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
from configs.utils import tasks as tasks_custom
from maze_lib.constants import max_reward, bonus_reward, reward_window

_FIXATION_THRESHOLD = 0.4
_FIXATION_STEPS = 25
_AGENT_POSITION = [0.5, 0.1]


class TrialInitialization():

    def __init__(self, stimulus_generator, prey_lead_in):
        self._stimulus_generator = stimulus_generator
        self._prey_lead_in = prey_lead_in

        self._prey_factors = dict(
            shape='circle', scale=0.015, c0=0.333, c1=1., c2=1.)
        self._agent_shape = 80*np.array([
            [0.50849388, 0.5], [0.50830827, 0.50176598], [0.50775955, 0.50345477], [0.50687169, 0.50499258], [0.5, 0.5],
            [0.49312831, 0.50499258], [0.49224045, 0.50345477], [0.49169173, 0.50176598], [0.49150612, 0.5],
            [0.49169173, 0.49823402], [0.49224045, 0.49654523], [0.49312831, 0.49500742], [0.49431648, 0.49368782], [0.49575306, 0.49264408],
            [0.49737525, 0.49192184], [0.49911215, 0.49155265], [0.50088785, 0.49155265], [0.50262475, 0.49192184], [0.50424694, 0.49264408],
            [0.50568352, 0.49368782], [0.50687169, 0.49500742], [0.50775955, 0.49654523], [0.50830827, 0.49823402], [0.50849388, 0.5]
        ])
        self._fixation_shape = 0.2 * np.array([
            [-5, 1], [-1, 1], [-1, 5], [1, 5], [1, 1], [5, 1], [5, -1], [1, -1],
            [1, -5], [-1, -5], [-1, -1], [-5, -1]
        ])

    def __call__(self):
        """State initializer."""
        stimulus = self._stimulus_generator()
        if stimulus is None:
            return None

        maze_size = stimulus['maze_size']
        prey_path = stimulus['prey_path']
        maze = maze_lib.Maze(maze_size, maze_size, prey_path=prey_path)
        maze.sample_distractors()
        tunnels = maze.to_sprites(wall_width=0.05, c0=0., c1=0., c2=0.5)

        prey = sprite.Sprite(**self._prey_factors)

        # pacman agent
        agent = sprite.Sprite(
            shape=self._agent_shape, scale=0.05,
            c0=0.1667, c1=1., c2=1)  # yellow in HSV
            # c0=0., c1=1., c2=0.66)  # red

        # Fixation cross and screen
        fixation = sprite.Sprite(
            x=0.5, y=0.5, shape=self._fixation_shape, scale=0.05,
            c0=0., c1=0., c2=0.)
        screen = sprite.Sprite(
            x=0.5, y=0.5, shape='square', scale=2., c0=0., c1=0., c2=1.)

        # Invisible eye sprite
        eye = sprite.Sprite(c0=0., c1=0., c2=0., opacity=0)

        state = collections.OrderedDict([
            ('maze_background', []),
            ('prey', [prey]),
            ('maze', tunnels),
            ('agent', [agent]),
            ('screen', [screen]),
            ('fixation', [fixation]),
            ('eye', [eye]),
        ])

        # Prey distance remaining is how far prey has to go to reach agent
        # It will be continually updated in the meta_state as the prey moves
        # prey_distance_remaining = (
        #     maze.arms[stimulus['prey_arm']].length + self._prey_lead_in)
        prey_distance_remaining = 10

        # TODO: add ts, tp here?
        self._meta_state = {
            'fixation_duration': 0,
            'motion_steps': 0,
            'phase': '',  # fixation -> offline -> motion -> online -> reward -> ITI
            'trial_name': '',
            'stimulus_features': stimulus['features'],
            'maze_arms': 17,
            'prey_arm': 17,
            'prey_distance_remaining': prey_distance_remaining,
            'correct_direction': None,
        }

        return state

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
                 delay_phase=True,
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
        self._delay_phase = delay_phase

        # Compute prey speed given ms_per_unit, assuming 60 fps
        self._prey_speed = 1000. / (60. * ms_per_unit) # 0.0083 frame width / ms
        self._agent_speed = 0.5 * self._prey_speed
        self._prey_lead_in = 0.07

        self._trial_init = TrialInitialization(
            stimulus_generator, prey_lead_in=self._prey_lead_in)

        # Create renderer
        self._observer = observers.PILRenderer(
            image_size=(256, 256),
            anti_aliasing=1,
            color_to_rgb='hsv_to_rgb',
        )

        self._construct_action_space()
        self._construct_game_rules()
        self._construct_physics()
        self._construct_task()

    def _construct_physics(self):
        """Construct physics."""
        self._maze_walk = maze_lib.MazeWalk(
            speed=0., avatar_layer='prey', start_lead_in=self._prey_lead_in)
        self._physics = physics_lib.Physics(
            # corrective_physics=[self._maze_walk],
        )

    def _construct_task(self):
        """Construct task."""
        prey_task = tasks_custom.TimeErrorReward(
            half_width=100,
            maximum=1,
            prey_speed=self._prey_speed,
        )
        timeout_task = tasks.Reset(
            condition=lambda _, meta_state: meta_state['phase'] == 'reward',
            steps_after_condition=15,
        )
        self._task = tasks.CompositeTask(prey_task, timeout_task)

    def _construct_action_space(self):
        """Construct action space."""

        controller_action_space = action_spaces_custom.GridRotate(
            action_layers='agent',
        )

        self._action_space = action_spaces.Composite(
            eye=action_spaces.SetPosition(action_layers=('eye',), inertia=0.),
            controller=controller_action_space,
        )

    def _construct_game_rules(self):
        """Construct game rules."""

        def _make_transparent(s):
            s.opacity = 0

        # 1. Fixation phase

        def _reset_physics(meta_state):
            # self._maze_walk.set_maze(
            #     meta_state['maze_arms'], meta_state['prey_arm'])
            self._maze_walk.speed = 0

        reset_physics = gr.ModifyMetaState(_reset_physics)

        def _should_increase_fixation_dur(state, meta_state):
            dist = np.linalg.norm(
                state['fixation'][0].position - state['eye'][0].position)
            eye_fixating = dist < _FIXATION_THRESHOLD
            agent = state['agent'][0]
            joystick_deflected = sum(agent.velocity != 0.)
            agent.position = _AGENT_POSITION
            return eye_fixating and not joystick_deflected

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
            return meta_state['fixation_duration'] >= _FIXATION_STEPS

        if not self._fixation_phase:
            fixation_duration = 10
        else:
            fixation_duration = np.inf

        phase_fixation = gr.Phase(
            one_time_rules=reset_physics,
            continual_rules=[increase_fixation_dur, reset_fixation_dur],
            end_condition=_should_end_fixation,
            duration=fixation_duration,
            name='fixation',
        )

        # 2. Offline phase

        disappear_fixation = gr.ModifySprites('fixation', _make_transparent)
        if self._delay_phase:
            delay_duration = lambda: np.random.randint(30, 60)
        else:
            delay_duration = 1
        phase_delay = gr.Phase(
            one_time_rules=disappear_fixation,
            duration=delay_duration,
            name='delay',
        )

        # 3. Motion phase

        disappear_screen = gr.ModifySprites('screen', _make_transparent)

        planning_duration = 10
        phase_planning = gr.Phase(
            one_time_rules=[disappear_screen],
            duration=planning_duration,
            name='planning',
        )

        # 4. Online phase

        def _unglue(meta_state):
            self._maze_walk.speed = self._prey_speed

        unglue = gr.ModifyMetaState(_unglue)

        def _update_motion_steps(meta_state):
            meta_state['motion_steps'] += 1
            meta_state['prey_distance_remaining'] -= self._prey_speed

        update_motion_steps = gr.ModifyMetaState(_update_motion_steps)

        def _end_motion_phase(state):
            agent = state['agent'][0]
            return agent.angle != 0

        phase_motion = gr.Phase(
            one_time_rules=unglue,
            continual_rules=update_motion_steps,
            end_condition=_end_motion_phase,
            name='motion',
        )

        # 5. Reward Phase

        reveal_prey = gr.ChangeLayer(
            old_layer='maze', new_layer='maze_background'
        )

        phase_reward = gr.Phase(
            one_time_rules=reveal_prey,
            continual_rules=update_motion_steps,
            name='reward',
        )

        # 6. ITI Phase

        # Final rules
        # fixation -> offline -> motion -> online -> reward -> ITI
        phase_sequence = gr.PhaseSequence(
            phase_fixation,
            phase_delay,
            phase_planning,
            phase_motion,
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
