"""Custom tasks."""

from moog import tasks
from dm_env import specs
import numpy as np
import inspect

_EPSILON = 1e-4


class ContactReward(tasks.AbstractTask):
    """ContactReward task.

    In this task if any sprite in layers_0 contacts any sprite in layers_1, a
    reward is given. Otherwise the reward is zero. Optionally, the task resets
    upon such a contact.

    This can be used for any contact-based reward, such as prey-seeking and
    predator-avoidance.
    """

    def __init__(self,
                 reward_fn,
                 layers_0,
                 layers_1,
                 condition=None,
                 reset_steps_after_contact=np.inf,
                 prey_opacity_staircase=None,
                 update_p_correct=None,
                 repeat_incorrect_trial=None
                 ):
        """Constructor.

        Args:
            reward_fn: Scalar or function (sprite_0, sprite_1) --> scalar. If
                function, sprite_0 and sprite_1 are sprites in layers_0 and
                layers_1 respectively.
            layers_0: String or iterable of strings. Reward is given if a sprite
                in this layer(s) contacts a sprite in layers_1.
            layers_1: String or iterable of strings. Reward is given if a sprite
                in this layer(s) contacts a sprite in layers_0.
            condition: Optional condition function. If specified, must have one
                of the following signatures:
                    * sprite_0, sprite_1 --> bool
                    * sprite_0, sprite_1, meta_state --> bool
                The bool is whether to apply reward for those sprites
                contacting.
            reset_steps_after_contact: Int. How many steps after a contact to
                reset the environment. Defaults to infinity, i.e. never
                resetting.
        """
        if not callable(reward_fn):
            self._reward_fn = lambda sprite_0, sprite_1: reward_fn
        else:
            self._reward_fn = reward_fn

        if not isinstance(layers_0, (list, tuple)):
            layers_0 = [layers_0]
        self._layers_0 = layers_0

        if not isinstance(layers_1, (list, tuple)):
            layers_1 = [layers_1]
        self._layers_1 = layers_1

        if condition is None:
            self._condition = lambda s_agent, s_target, meta_state: True
        elif len(inspect.signature(condition).parameters.values()) == 2:
            self._condition = lambda s_a, s_t, meta_state: condition(s_a, s_t)
        else:
            self._condition = condition

        self._reset_steps_after_contact = reset_steps_after_contact

        # custom staircase
        self._prey_opacity_staircase = prey_opacity_staircase
        self._update_p_correct = update_p_correct
        self._repeat_incorrect_trial = repeat_incorrect_trial

    def reset(self, state, meta_state):
        self._steps_until_reset = np.inf

    def reward(self, state, meta_state, step_count):
        """Compute reward.

        If any sprite_0 in self._layers_0 overlaps any sprite_1 in
        self._layers_1 and if self._condition(sprite_0, sprite_1, meta_state) is
        True, then the reward is self._reward_fn(sprite_0, sprite_1).

        Args:
            state: OrderedDict of sprites. Environment state.
            meta_state: Environment state. Unconstrained type.
            step_count: Int. Environment step count.

        Returns:
            reward: Scalar reward.
            should_reset: Bool. Whether to reset task.
        """
        reward = 0
        sprites_0 = [s for k in self._layers_0 for s in state[k]]
        sprites_1 = [s for k in self._layers_1 for s in state[k]]
        for s_0 in sprites_0:
            for s_1 in sprites_1:
                if not self._condition(s_0, s_1, meta_state):
                    continue
                if s_0.overlaps_sprite(s_1):
                    reward = self._reward_fn(s_0, s_1)
                    if self._steps_until_reset == np.inf:
                        self._steps_until_reset = (
                            self._reset_steps_after_contact)
                    # custom staircase
                    if self._update_p_correct is not None:
                        self._update_p_correct.step(reward, meta_state['num_junctions'],
                                                    meta_state['num_amb_junctions'])
                    if self._repeat_incorrect_trial is not None:
                        self._repeat_incorrect_trial.step(reward)
                    if self._prey_opacity_staircase is not None:
                        self._prey_opacity_staircase.step(reward)

        self._steps_until_reset -= 1
        should_reset = self._steps_until_reset < 0

        return reward, should_reset


class TimeErrorReward(tasks.AbstractTask):
    """Timing task.

    Reward is a tooth function around the time the prey exits the maze in the
    direction of the response.
    """

    def __init__(self,
                 half_width,
                 maximum,
                 prey_speed,
                 max_rewarding_dist,
                 prey_opacity_staircase,
                 response_layer='agent',
                 prey_layer='prey'):
        """Constructor.

        Args:
            half_width: reward window (i.e., width of tooth function)
            maximum: maximum reward at zero time error
            prey_speed: time error is computed by dividing distance_remaining with prey_speed
            max_rewarding_dist: Scalar. Maximum distance (in units of screen
                width) from the correct exit to give reward.
            response_layer: sprite layer
            prey_layer: sprite layer

        """
        self._half_width = half_width
        self._maximum = maximum
        self._prey_speed = prey_speed
        self._max_rewarding_dist = max_rewarding_dist
        self._response_layer = response_layer
        self._prey_layer = prey_layer
        self._prey_opacity_staircase = prey_opacity_staircase

    def reset(self, state, meta_state):
        del state
        del meta_state
        self._reward_given = False

    def _tooth_function(self, speed, distance_remaining):
        time_remaining = distance_remaining / speed
        time_error = np.abs(time_remaining)
        slope = self._maximum / self._half_width
        reward = self._maximum - time_error * slope
        reward = max(reward, 0)
        return reward

    def reward(self, state, meta_state, step_count):
        del step_count

        if meta_state['phase'] == 'reward' and not self._reward_given:  # and state['agent'][0].metadata['response']
            # Update reward

            agent = state['agent'][0]
            id_vertical = np.mod(meta_state['correct_side'], 2)  # 0 for 0/2 (bottom/top)
            prey_exit_x = meta_state['prey_path'][-1][0]
            error_x = agent.x - prey_exit_x
            prey_exit_y = meta_state['prey_path'][-1][1]
            error_y = agent.y - prey_exit_y
            agent_prey_dist = np.abs((1 - id_vertical) * error_x + id_vertical * error_y)

            if agent_prey_dist < self._max_rewarding_dist:
                reward = self._tooth_function(
                    self._prey_speed, meta_state['prey_distance_remaining'])
            else:
                # Agent is too far away from prey exit in the x axis
                reward = 0
            self._reward_given = True

            if self._prey_opacity_staircase is not None:
                self._prey_opacity_staircase.step(reward)
        else:
            reward = 0

        return reward, False




class OfflineReward(tasks.AbstractTask):
    """Give reward if agent stops at correct exit during offline phase."""

    def __init__(self,
                 phase,
                 max_rewarding_dist=0.,
                 path_prey_opacity_staircase=None,
                 path_prey_position_staircase=None,
                 update_p_correct=None,
                 repeat_incorrect_trial=None):
        """Constructor.
        
        Args:
            phase: String. Phase of task in which to give reward.
            max_rewarding_dist: Scalar. Maximum distance (in units of screen
                width) from the correct exit to give reward. The reward is
                linearly interpolated between zero at this value and 1 at the
                correct exit.
        """
        self._phase = phase
        self._max_rewarding_dist = max_rewarding_dist
        self._path_prey_opacity_staircase = path_prey_opacity_staircase
        self._path_prey_position_staircase = path_prey_position_staircase
        self._update_p_correct = update_p_correct
        self._repeat_incorrect_trial = repeat_incorrect_trial

    def reset(self, state, meta_state):
        del state
        del meta_state
        self._reward_given = False

    def reward(self, state, meta_state, step_count):
        del step_count
        if len(state['agent']) > 0:
            agent = state['agent'][0]
            if (meta_state['phase'] == self._phase and
                    not self._reward_given and
                    agent.metadata['moved_h'] and
                    np.all(state['agent'][0].velocity == 0)):
                id_vertical = np.mod(meta_state['correct_side'], 2)  # 0 for 0/2 (bottom/top)
                prey_exit_x = meta_state['prey_path'][-1][0]
                error_x = agent.x - prey_exit_x
                prey_exit_y = meta_state['prey_path'][-1][1]
                error_y = agent.y - prey_exit_y
                agent_prey_dist = np.abs((1 - id_vertical) * error_x + id_vertical * error_y)
                meta_state['offline_error'] = agent_prey_dist
                meta_state['end_x_agent'] = agent.x
                meta_state['end_y_agent'] = agent.y
                reward = max(0, 1 - agent_prey_dist / (self._max_rewarding_dist + _EPSILON))
                self._reward_given = True

                if self._path_prey_opacity_staircase is not None:
                    self._path_prey_opacity_staircase.step(reward)
                if self._path_prey_position_staircase is not None:
                    self._path_prey_position_staircase.step(reward)
                if self._update_p_correct is not None:
                    self._update_p_correct.step(reward,meta_state['num_junctions'],meta_state['num_amb_junctions'])
                if self._repeat_incorrect_trial is not None:
                    self._repeat_incorrect_trial.step(reward)
            else:
                reward = 0.
        else:
            reward = 0.
        
        return reward, False


class BeginPhase(tasks.AbstractTask):
    """Task to give reward at beginning of phase."""

    def __init__(self, phase):
        """Constructor."""
        self._phase = phase

    def reset(self, state, meta_state):
        del state
        del meta_state
        self._reward_given = False

    def reward(self, state, meta_state, step_count):
        del state
        del step_count

        if meta_state['phase'] == self._phase and not self._reward_given:
            self._reward_given = True
            return 0, False
        else:
            return 0, False