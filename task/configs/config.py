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

_IMAGE_SIZE = [10,20,30]

class TrialInitialization():

    def __init__(self, stimulus_generator, prey_lead_in, border_width):
        self._stimulus_generator = stimulus_generator
        self._prey_lead_in = prey_lead_in
        self._border_width = border_width

        self._prey_factors = dict(
            shape='circle', scale=0.015, c0=0.333, c1=1., c2=1.)
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
        maze.sample_distractor_exit(prey_path=prey_path)
        maze.sample_distractors()
        tunnels = maze.to_sprites(
            wall_width=0.05, border_width=self._border_width, c0=0., c1=0.,
            c2=0.5)

        prey = sprite.Sprite(**self._prey_factors)

        # Response sprites
        response_x = [0., 1., 0.5, 0.5]
        response_y = [0.5, 0.5, 0., 1.]
        response_angle = [0.5 * np.pi, 0.5 * np.pi, 0., 0.]
        responses = [
            sprite.Sprite(
                x=x, y=y, shape='square', aspect_ratio=0.15, angle=a,
                c0=0.667, c1=1., c2=1, opacity=0)
            for x, y, a in zip(response_x, response_y, response_angle)
        ]

        # Fixation cross and screen
        fixation = sprite.Sprite(
            x=0.5, y=0.5, shape=self._fixation_shape, scale=0.05,
            c0=0., c1=0., c2=0., opacity=0)
        screen = sprite.Sprite(
            x=0.5, y=0.5, shape='square', scale=2., c0=0., c1=0., c2=1.)

        # Invisible eye sprite
        eye = sprite.Sprite(c0=0., c1=0., c2=0., opacity=0)

        state = collections.OrderedDict([
            ('maze_background', []),
            ('prey', [prey]),
            ('maze', tunnels),
            ('response', responses),
            ('screen', [screen]),
            ('fixation', [fixation]),
            ('eye', [eye]),
        ])

        # Prey distance remaining is how far prey has to go to exit maze
        # It will be continually updated in the meta_state as the prey moves
        prey_distance_remaining = (self._prey_lead_in +
            (1 - 2 * self._border_width) * len(prey_path) / maze_size)

        # randomly choose image size across trials
        image_size = np.random.choice(_IMAGE_SIZE)

        # TODO: add ts, tp here?
        self._meta_state = {
            'fixation_duration': 0,
            'motion_steps': 0,
            'phase': '',  # fixation -> offline -> motion -> online -> reward -> ITI
            'trial_name': '',
            'stimulus_features': stimulus['features'],
            'prey_path': prey_path,
            'maze_size': maze_size,
            'image_size': image_size,
            'prey_distance_remaining': prey_distance_remaining,
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
                 offline_phase=True,
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
        self._offline_phase = offline_phase

        # Compute prey speed given ms_per_unit, assuming 60 fps
        self._prey_speed = 1000. / (60. * ms_per_unit) # 0.0083 frame width / ms
        self._prey_lead_in = 0.15
        self._border_width = 0.18  # boundary space around the maze on all sides

        self._trial_init = TrialInitialization(
            stimulus_generator, prey_lead_in=self._prey_lead_in,
            border_width=self._border_width)

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
            corrective_physics=[self._maze_walk],
        )

    def _construct_task(self):
        """Construct task."""

        prey_task = tasks_custom.TimeErrorReward(
            half_width=20,  # given 60 Hz, 333 ms
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
        self._action_space = action_spaces.Composite(
            eye=action_spaces.SetPosition(action_layers=('eye',), inertia=0.),
            hand=action_spaces_custom.CardinalDirections('response'),
        )

    def _construct_game_rules(self):
        """Construct game rules."""

        def _make_transparent(s):
            s.opacity = 0

        def _make_opaque(s):
            s.opacity=255

        # 1. ITI phase

        def _reset_physics(meta_state):
            self._maze_walk.set_prey_path(
                meta_state['prey_path'], meta_state['maze_size'],
                self._border_width)
            self._maze_walk.speed = 0

        reset_physics = gr.ModifyMetaState(_reset_physics)

        phase_iti = gr.Phase(
            one_time_rules=reset_physics,
            duration=10,
            name='iti',
        )

        # 2. Fixation phase

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
            return meta_state['fixation_duration'] >= _FIXATION_STEPS

        if not self._fixation_phase:
            fixation_duration = 10
        else:
            fixation_duration = np.inf

        appear_fixation = gr.ModifySprites('fixation', _make_opaque)

        phase_fixation = gr.Phase(
            one_time_rules=appear_fixation,
            continual_rules=[increase_fixation_dur, reset_fixation_dur],
            end_condition=_should_end_fixation,
            duration=fixation_duration,
            name='fixation',
        )

        # 3. Offline phase
        def _end_offline_phase(state):
            return np.any([s.opacity > 0 for s in state['response']])

        disappear_fixation = gr.ModifySprites('fixation', _make_transparent)
        disappear_screen = gr.ModifySprites('screen', _make_transparent)

        phase_offline = gr.Phase(
            one_time_rules=[disappear_fixation, disappear_screen],
            name='offline',
            end_condition=_end_offline_phase,
        )

        # 4. Visible motion phase

        def _unglue(meta_state):
            self._maze_walk.speed = self._prey_speed

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

        # 5. Invisible motion phase

        def _end_motion_phase(state):
            return np.any([s.opacity > 0 for s in state['response']])

        hide_prey = gr.ModifySprites('prey', _make_transparent)

        phase_motion_invisible = gr.Phase(
            one_time_rules=hide_prey,
            continual_rules=update_motion_steps,
            end_condition=_end_motion_phase,
            name='motion_invisible',
        )

        # 6. Reward Phase

        reveal_prey = gr.ModifySprites('prey', _make_opaque)

        phase_reward = gr.Phase(
            one_time_rules=reveal_prey,
            continual_rules=update_motion_steps,
            name='reward',
        )

        # Final rules
        # fixation -> offline -> visible motion -> invisible motion -> reward -> ITI
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
