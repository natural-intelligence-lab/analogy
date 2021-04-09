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

_FIXATION_THRESHOLD = 0.4
_FIXATION_STEPS = 25


class TrialInitialization():

    def __init__(self, stimulus_generator, maze_scale=0.03):
        self._stimulus_generator = stimulus_generator
        self._maze_scale = maze_scale

        self._agent_factors = dict(
            x=0.5, y=0.5, shape='square', scale=0.05, c0=0., c1=1., c2=0.66)
        self._prey_factors = dict(
            x=0.5, y=0.5, shape='circle', scale=0.015, c0=0.333, c1=1., c2=1.)
        
        self._fixation_shape = 0.2 * np.array([
            [-5, 1], [-1, 1], [-1, 5], [1, 5], [1, 1], [5, 1], [5, -1], [1, -1],
            [1, -5], [-1, -5], [-1, -1], [-5, -1]
        ])

    def __call__(self):
        """State initializer."""
        stimulus = self._stimulus_generator()
        if stimulus is None:
            return None

        maze_arms = stimulus['maze_arms']
        maze_arms = [
            [(np.array(d), self._maze_scale * l) for d, l in arm]
            for arm in stimulus['maze_arms']
        ]
        maze = maze_lib.Maze(maze_arms)
        tunnels = maze.to_sprites(arm_width=0.01, c0=0., c1=0., c2=0.5)

        prey = sprite.Sprite(**self._prey_factors)
        agent = sprite.Sprite(**self._agent_factors)

        # Fixation cross and sreen
        fixation = sprite.Sprite(
            x=0.5, y=0.5, shape=self._fixation_shape, scale=0.05,
            c0=0., c1=0., c2=0.)
        screen = sprite.Sprite(
            x=0.5, y=0.5, shape='square', scale=2., c0=0., c1=0., c2=1.)

        # Invisible eye sprite
        eye = sprite.Sprite(c0=0., c1=0., c2=0., opacity=0)

        state = collections.OrderedDict([
            ('prey', [prey]),
            ('maze', tunnels),
            ('agent', [agent]),
            ('screen', [screen]),
            ('fixation', [fixation]),
            ('eye', [eye]),
        ])

        self._meta_state = {
            'fixation_duration': 0,
            'motion_steps': 0,
            'phase': '',
            'trial_name': '',
            'stimulus_features': stimulus['features'],
            'maze_arms': maze_arms,
            'prey_arm': stimulus['prey_arm'],
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
        """
        self._stimulus_generator = stimulus_generator
        self._fixation_phase = fixation_phase
        self._delay_phase = delay_phase

        # Compute prey speed given ms_per_unit
        self._prey_speed = 1000. / (60. * ms_per_unit)
        self._agent_speed = 0.5 * self._prey_speed

        self._trial_init = TrialInitialization(stimulus_generator)

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
        self._maze_walk = maze_lib.MazeWalk(speed=0., avatar_layer='prey')
        self._physics = physics_lib.Physics(
            corrective_physics=[self._maze_walk])

    def _construct_task(self):
        """Construct task."""
        prey_task = tasks.ContactReward(
            reward_fn=lambda s_agent, s_prey: 1,
            layers_0='agent',
            layers_1='prey',
            reset_steps_after_contact=10,
        )
        reset_task = tasks.Reset(
            condition=lambda _, meta_state: meta_state['phase'] == 'response',
            steps_after_condition=150,
        )
        self._task = tasks.CompositeTask(prey_task, reset_task)

    def _construct_action_space(self):
        """Construct action space."""

        controller_action_space = action_spaces.Grid(
            scaling_factor=self._agent_speed,
            action_layers='agent',
            control_velocity=True,
        )

        self._action_space = action_spaces.Composite(
            eye=action_spaces.SetPosition(action_layers=('eye',), inertia=0.),
            controller=controller_action_space,
        )

        self._action_space = controller_action_space

    def _construct_game_rules(self):
        """Construct game rules."""
        
        def _make_transparent(s):
            s.opacity = 0

        # Fixation phase

        def _reset_physics(meta_state):
            self._maze_walk.set_maze(
                meta_state['maze_arms'], meta_state['prey_arm'])
            self._maze_walk._speed = 0
        reset_physics = gr.ModifyMetaState(_reset_physics)

        def _should_increase_fixation_dur(state, meta_state):
            dist = np.linalg.norm(
                state['fixation'][0].position - state['eye'][0].position)
            eye_fixating = dist < _FIXATION_THRESHOLD
            agent = state['agent'][0]
            joystick_deflected = sum(agent.velocity != 0.)
            agent.position = [0.5, 0.5]
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

        # Delay phase

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

        # Planning phase

        disappear_screen = gr.ModifySprites('screen', _make_transparent)

        planning_duration = 10
        phase_planning = gr.Phase(
            one_time_rules=[disappear_screen],
            duration=planning_duration,
            name='planning',
        )

        # Response phase

        def _unglue(meta_state):
            self._maze_walk._speed = self._prey_speed
        unglue = gr.ModifyMetaState(_unglue)

        def _update_motion_steps(meta_state):
            meta_state['motion_steps'] += 1
        update_motion_steps = gr.ModifyMetaState(_update_motion_steps)

        def _end_motion_phase(state):
            return len(state['prey']) == 0
        phase_response = gr.Phase(
            one_time_rules=unglue,
            continual_rules=update_motion_steps,
            name='response',
        )

        # Final rules

        phase_sequence = gr.PhaseSequence(
            phase_fixation,
            phase_delay,
            phase_planning,
            phase_response,
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
    