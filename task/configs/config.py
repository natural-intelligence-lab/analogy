"""Common grid_chase task config.

2022/5/10
1) % correct initial direction (# junctions): agent.metadata['id_left0'] & meta_state['id_left0'] > meta_state['id_correct_offline']
2) ball size/reward decreasing after glued at bottom
3) slow down (color change to red) after initial movement error
4) increase wall size
5) highlight path

2022/5/6
1) ball moving right after paddle moves; paddle freely moving
2) plan: [0 turn 0]>[0 1](0,1 junctions)>[0 2]... (# turns, # junctions, # distractors)


old-TBD
1) Control trials: highlight path before
2) highlight during online
3) 2-turn, 4-turn (to fight against 2-turn bias)

2022/5/2
1) repeat error trial: _ID_REPEAT_INCORRECT_TRIAL
2) black background
3) moving ball at the beginning

2022/4/30
everything same as H (monkeyEphys) except path aid staircase
- new features: no gap, 0-turn with no overlap constraint (new maze), paddle initially at middle & booster
1) Impose 500ms for maze-on

2022/4/21
1) make maze on & path prey optional (to resume staircase,
    - set _PATH_PREY_DURATION to np.inf
    - (_OPACITY_INIT_>0) and
    - bring back highlight_path in continual_rules
2) clean up code: rename task phase & meta_state
3) TBD: streamline eye data collection & distractor_path in samplers.py

2022/4/4
1) add gap for junctions between distractor paths
2) implement slower paddle around target
3) plot_tp_ts: get % correct for fully invisible trials

2022/3/29
1) implement gap
2) (TBD) magnet for response

2022/3/7 debug list for wire_maze
1) (done) when last segment is short, ball dynamics is weird
2) loop?
3) why so fast together with monkey_train branch?

2022/2/25
1) remove online for training (for now): offline_timeout_task
2) vertical movement only (exit on the left/right only): _step_num_turns in layered.py
3) increase paddle size

2022/2/21
1) exit on left or right: glue_path_prey
2) put paddle corners (allowing # turns to be 1 to 4)

2022/1/21
1) remove offline simuation period ("foreperiod") to measure RT

changes @ 2021/11/21
1) implement movie for highlighting maze path
2) random agent location (agent_x0)
3) remove prey only period (_FAKE_PREY_DURATION=0)

TO DO:
1) for offline+online (H), no online if no offline reward
2) how to change _MAX_REWARDING_DIST during task running? _MAX_REWARDING_DIST
3) change initial agent position? configs/levels/training/vertical_timing.py

"""

################################################################################################
import abc
import collections
import math
import numpy as np
import copy

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
from configs.utils import physics as physics_custom
from configs.utils import tasks_offline as tasks_custom_offline
from maze_lib.constants import max_reward, bonus_reward, reward_window

################################################################################################
# stimlulus
_AGENT_Y = 0.05
_MAZE_Y = 0.15
_MAZE_WIDTH = 0.7
_IMAGE_SIZE = [24]  # [8, 16, 24]
_AGENT_SCALE = 0.1 # 0.15  # 0.03  # 0.05 # 0.10  # 0.15
_AGENT_ASPECT_RATIO = 0.2 # 4 # 8 # 4
_PREY_SCALE = 0.03
_WALL_WIDTH = 12*0.05
# _MIN_DIST_AGENT = 0.1 # /2  # minimum distance between initial agent position and target exit
# _MAX_DIST_AGENT = 0.3 # 0.5
_P_PREY0 = 0 # 0.1  #  0.3 # 0.5   # 0.9  # prey's initial position as % of path
_GAIN_PATH_PREY = 1  # 2.5 # 2 # 3  # 1  # speed gain for path prey
_PATH_PREY_OPACITY = 120  # 50
_GAIN_SLOW_OFFLINE_ERROR = 0.3 # after offline error, prey_speed is scaled by this factor

_ID_REPEAT_INCORRECT_TRIAL = True

# fixation
_FIXATION_THRESHOLD = 0.4

# time
_ITI = 60
_FIXATION_STEPS = 0 # 60  # 30
_BALL_ON_DURATION=30 # 500ms # 20 # 333 ms # 0 # 30 # 500ms
_MAZE_ON_DURATION=30  # 30 # 60 # 30 # 60 # 1s
_PATH_PREY_DURATION=0 # np.inf # 0

_MAX_WAIT_TIME_GAIN = 10 # 2 # when tp>2*ts, abort
# _JOYSTICK_FIXATION_POSTOFFLINE = 36 # 600

# reward
_MAX_REWARDING_DIST=((_AGENT_SCALE)/2)+(_PREY_SCALE/2) # =((_AGENT_ASPECT_RATIO*_AGENT_SCALE)/2)+(_PREY_SCALE/2)   #/3*2 # _AGENT_SCALE/2 # 0.15 # also scale of agent sprite
_EPSILON=1e-4 # FOR REWARD FUNCTION
_REWARD = 3 # 6 # 100 ms # post zero prey_distance
_TOOTH_HALF_WIDTH = 40 # 60 # 40 # 666ms

# joystick slowing near potential exits
_SLOWING_DIST=((_AGENT_SCALE)/2)+(_PREY_SCALE/2)
_GAIN_MASS = 1.5
_DEFAULT_MASS = 1 # 0.5
_ACTION_SCALING_FACTOR = 0.01 # 0.015 # 0.01

# staircase
# _staircase for prey (online)
_STEP_OPACITY_UP = 0 # 5 ## 5 # 10 #      0 # 1 # 2021/9/8 # 1 # 0 # 1 # 2 # 3 # 10  # [0 255] # 2021/9/3
_STEP_OPACITY_DOWN = 0 # 10 #     5 # 30 # 40  # [0 255]
_OPACITY_INIT = 255 # 100  # 20 # 100 # 20 # 20 # 100 #     10 # 100 # 10
_DIM_DURATION = 2 # [sec]

# staircase for path prey (offline)
_STEP_OPACITY_UP_ = 0 # 10 # 1  # 0 # 5 # 10 #      0 # 1 # 2021/9/8 # 1 # 0 # 1 # 2 # 3 # 10  # [0 255] # 2021/9/3
_STEP_OPACITY_DOWN_ = 0 # 10 #     5 # 30 # 40  # [0 255]
_OPACITY_INIT_ = 100 # 0 # 100 # 200 # 100 # 20  # 0 # 20 # 100 #     10 # 100 # 10
_P_DIM_DISTANCE_ = 0 # 2/3
_DIM_DURATION_ = 2 # [sec]

# staircase p(visible aid)
_P_PATHPREY_DIM_DISTANCE = 0 # 2/3 # 1/2 # 1/3 # 1/2 # 1/4 # 0 # 1/3 # 1/2 # 1/3 # 1/2 # 0 # 2/3
_PathPreyPosition_INIT_ = 1 # 2/3 # 1 # 0 # 1/2
_STEP_PathPreyPosition_DOWN_ = 0 # 0.1 # 1/4  # 0.1
_STEP_PathPreyPosition_UP_ = 0 # 0.1 # 1/4  # 0.1

# performance monitoring
_MAX_NUM_TURNS = 2 # should be match wire_maze.py
_N_GRID = 16
_MAX_NUM_JUNCTION = (_N_GRID-2)*_MAX_NUM_TURNS
_n_correct_n_junction = np.zeros(_MAX_NUM_JUNCTION)
_n_correct_n_amb_junction = np.zeros(_MAX_NUM_JUNCTION)
_n_trial = np.zeros(_MAX_NUM_JUNCTION)
_n_trial_amb = np.zeros(_MAX_NUM_JUNCTION)

################################################################################################
class UpdatePercentCorrect():
    def __init__(self,
                 n_correct_n_junction=_n_correct_n_junction,
                 n_correct_n_amb_junction=_n_correct_n_amb_junction,
                 n_trial=_n_trial,
                 n_trial_amb=_n_trial_amb):
        self._n_correct_n_junction = n_correct_n_junction
        self._n_correct_n_amb_junction = n_correct_n_amb_junction
        self._n_trial = n_trial
        self._n_trial_amb = n_trial_amb

    def step(self, reward,num_junctions, num_amb_junctions,id_correct_offline):
        self._n_trial[num_junctions] += 1
        self._n_trial_amb[num_amb_junctions] += 1
        if id_correct_offline>0:  # reward > 0:
            self._n_correct_n_junction[num_junctions] += 1
            self._n_correct_n_amb_junction[num_amb_junctions] += 1
    @property
    def n_trial(self):
        return self._n_trial
    @property
    def n_trial_amb(self):
        return self._n_trial_amb
    @property
    def n_correct_n_amb_junction(self):
        return self._n_correct_n_amb_junction
    @property
    def n_correct_n_junction(self):
        return self._n_correct_n_junction

class PreyOpacityStaircase():
    def __init__(self,
                 init_value=_OPACITY_INIT,
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

class PathPreyOpacityStaircase():
    def __init__(self,
                 init_value=_OPACITY_INIT_,
                 success_delta=_STEP_OPACITY_DOWN_,
                 failure_delta=_STEP_OPACITY_UP_,
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

class PathPreyPositionStaircase():
    def __init__(self,
                 init_value=_PathPreyPosition_INIT_,
                 failure_delta=_STEP_PathPreyPosition_DOWN_,
                 success_delta=_STEP_PathPreyPosition_UP_,
                 minval=0,
                 maxval=1):
        self._path_prey_position = init_value
        self._success_delta = success_delta
        self._failure_delta = failure_delta
        self._minval = minval
        self._maxval = maxval
    def step(self, reward):
        if reward <= 0:
            self._path_prey_position = max(self._path_prey_position - self._failure_delta, self._minval)
        elif reward > 0: # higher with reward
            self._path_prey_position = min(self._path_prey_position + self._success_delta, self._maxval)
    @property
    def path_prey_position(self):
        return self._path_prey_position

class RepeatIncorrectTrial():
    def __init__(self,
                 id_repeat_incorrect_trial=_ID_REPEAT_INCORRECT_TRIAL,
                 id_correct_offline0=True):
        self._id_correct_offline=id_correct_offline0
        self._id_repeat_incorrect_trial=id_repeat_incorrect_trial
        self._stimulus = None

    def step(self, reward):
        if self._id_repeat_incorrect_trial:
            if reward <= 0:
                self._id_correct_offline = False
            elif reward > 0:
                self._id_correct_offline = True
        else:
            self._id_correct_offline = True
    @property
    def id_correct_offline(self):
        return self._id_correct_offline
    @property
    def stimulus(self):
        return self._stimulus
    @stimulus.setter
    def stimulus(self, value):
        self._stimulus = value


################################################################################################
class TrialInitialization():

    def __init__(self, stimulus_generator, prey_lead_in, prey_speed, static_prey=False,
                 static_agent=False,prey_opacity_staircase=None,path_prey_opacity_staircase=None,
                 path_prey_position_staircase=None,update_p_correct=None,repeat_incorrect_trial=None,ms_per_unit=None):
        self._stimulus_generator = stimulus_generator
        self._prey_lead_in = prey_lead_in
        self._prey_speed = prey_speed
        self._static_prey = static_prey
        self._static_agent = static_agent
        self._prey_opacity_staircase=prey_opacity_staircase
        self._path_prey_opacity_staircase = path_prey_opacity_staircase
        self._path_prey_position_staircase = path_prey_position_staircase
        self._update_p_correct = update_p_correct
        self._repeat_incorrect_trial = repeat_incorrect_trial
        self._prey_factors = dict(
            shape='circle', scale=_PREY_SCALE, c0=0, c1=255, c2=0) # scale=0.015, c0=0, c1=255, c2=0)
        self._fixation_shape = 0.2 * np.array([
            [-5, 1], [-1, 1], [-1, 5], [1, 5], [1, 1], [5, 1], [5, -1], [1, -1],
            [1, -5], [-1, -5], [-1, -1], [-5, -1]
        ])
        self._ms_per_unit = ms_per_unit

    def __call__(self):
        """State initializer."""
        if self._repeat_incorrect_trial.id_correct_offline:
            stimulus = self._stimulus_generator()
            self._repeat_incorrect_trial.stimulus = copy.deepcopy(stimulus)  # save stimulus to repeat later
        else:
            stimulus = copy.deepcopy(self._repeat_incorrect_trial.stimulus)

        if stimulus is None:
            return None

        maze_width = stimulus['maze_width']
        maze_height = stimulus['maze_height']
        maze_walls = stimulus['maze_walls']
        num_turns = stimulus['features']['num_turns']
        path_walls = stimulus['features']['maze_prey_walls']  # list
        x_junction = [x / stimulus['features']['path_length'] for x in stimulus['features']['x_overlap']]
        maze = maze_lib.Maze(maze_width, maze_height, all_walls=maze_walls)
        cell_size = _MAZE_WIDTH / maze_width
        tunnels = maze.to_sprites(
            wall_width=_WALL_WIDTH, cell_size=cell_size, bottom_border=_MAZE_Y, c0=128,
            c1=128, c2=128)
        # to highlight path touched by path aid
        maze_prey_walls = maze_lib.Maze(maze_width, maze_height, all_walls=path_walls)
        path_wall_sprite = maze_prey_walls.to_sprites(
            wall_width=_WALL_WIDTH, cell_size=cell_size, bottom_border=_MAZE_Y, c0=32,c1=128, c2=32, opacity=0)  # green

        # Compute scaled and translated prey path
        prey_path = 0.5 + np.array(stimulus['prey_path'])
        cell_size = _MAZE_WIDTH / maze_width
        prey_path *= cell_size
        total_width = cell_size * maze_width
        prey_path += np.array([[0.5 * (1 - total_width), _MAZE_Y]])

        # identify exits
        wall_array = np.asarray(maze_walls)
        x_exit = wall_array[np.argwhere(wall_array==-0.5)[:,0],0,0]  # in maze wall coordinate
        x_exit *= cell_size
        x_exit += 0.5 * (1 - total_width)

        # Compute scaled and translated distractor path
        if 'distractor_path' in stimulus['features']:
            if stimulus['features']['distractor_path']!=[]:
                distractor_path = 0.5 + np.array(stimulus['features']['distractor_path'])
                distractor_path *= cell_size
                distractor_path += np.array([[0.5 * (1 - total_width), _MAZE_Y]])
            else:
                distractor_path = []
        else:
            distractor_path = []

        prey = sprite.Sprite(**self._prey_factors)

        if self._static_prey:
            prey.position = [prey_path[0][0], _AGENT_Y - 0.001]

        # controlling initial prey position
        _P_PREY = np.int(prey_path.shape[0]*_P_PREY0) # _P_PREY0 = 0
        prey_path = prey_path[_P_PREY:]

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
        # set up agent initial locations
        correct_side = 0
        _agent_y0 = _AGENT_Y
        _agent_x0 = 0.5  # np.random.rand()
        # while np.abs(_agent_x0-prey_path[-1][0]) < _MIN_DIST_AGENT or np.abs(_agent_x0-prey_path[-1][0]) > _MAX_DIST_AGENT:
        #     _agent_x0 = np.random.rand()  # if too close, resample

        # fake prey
        distance_to_start = (_BALL_ON_DURATION/60) / (self._ms_per_unit/1000)   # 30/60 [s] / 2[s/unit] = 0.25
        direction_fake_prey = np.array([np.round(np.random.rand())*2-1, 0])
        speed_fake_prey = (1/60) / (self._ms_per_unit/1000)
        fake_prey = sprite.Sprite(shape='circle', scale=_PREY_SCALE, c0=0, c1=255, c2=0, opacity=0,
                                  metadata={'distance_to_start': distance_to_start, 'direction_fake_prey': direction_fake_prey,'speed':speed_fake_prey})
        fake_prey.position = [prey_path[0][0] -direction_fake_prey[0]*distance_to_start,
                              prey_path[0][1]+self._prey_lead_in]

        state = collections.OrderedDict([
            ('agent', []),
            ('maze', tunnels),
            ('prey_wall', path_wall_sprite),
            ('prey', [prey]),
            ('screen', [screen]),
            ('joystick_fixation', [joystick_fixation]),
            ('joystick', [joystick]),
            ('fixation', [fixation]),
            ('eye', [eye]),
            ('fake_prey', [fake_prey]),
            ('prey_path', []),
            ('path_prey', []),
        ])

        # Prey distance remaining is how far prey has to go to reach agent
        # It will be continually updated in the meta_state as the prey moves
        prey_distance_remaining = (
            self._prey_lead_in + cell_size * len(prey_path) + _MAZE_Y -
            _AGENT_Y)

        # randomly choose image size across trials
        image_size = np.random.choice(_IMAGE_SIZE)

        # opacity staircase
        if self._prey_opacity_staircase is None:
            self._prey_opacity = 255
        else:
            self._prey_opacity = self._prey_opacity_staircase.opacity
        if self._path_prey_opacity_staircase is None:
            self._path_prey_opacity = _PATH_PREY_OPACITY
        else:
            self._path_prey_opacity = self._path_prey_opacity_staircase.opacity

        # position path aid
        if self._path_prey_position_staircase is None:
            self._path_prey_position = _P_PATHPREY_DIM_DISTANCE  # default: fully visible
        else:
            self._path_prey_position = self._path_prey_position_staircase.path_prey_position

        n_correct_n_junction = self._update_p_correct.n_correct_n_junction
        n_correct_n_amb_junction = self._update_p_correct.n_correct_n_amb_junction
        n_trial = self._update_p_correct.n_trial
        n_trial_amb = self._update_p_correct.n_trial_amb

        # to make data json-seriziable
        del stimulus['features']['maze_prey_walls']

        self._meta_state = {
            'fix_dur': 0,
            'motion_steps': 0,
            'phase': '',  # fixation -> offline -> motion -> online -> reward -> ITI
            'stimulus_features': stimulus['features'],
            'prey_path': np.around(prey_path, decimals=3),
            'prey_speed': self._prey_speed,
            'prey_opacity': self._prey_opacity,
            'path_prey_opacity': self._path_prey_opacity,
            'half_width' : _TOOTH_HALF_WIDTH,
            'maze_width': maze_width,
            'maze_height': maze_height,
            'image_size': image_size,
            'prey_distance_remaining': prey_distance_remaining,
            'prey_distance_invisible': cell_size * len(prey_path) + _MAZE_Y - _AGENT_Y,
            'prey_distance': cell_size * len(prey_path),
            'slope_opacity': 0,
            'RT_offline': 0,
            'tp': 0,
            'ts': 0,
            'max_rewarding_dist': _MAX_REWARDING_DIST,
            'joy_fix_post': 0,
            'num_turns': num_turns,
            'end_x_agent': prey_path[-1][0],
            'end_y_agent': prey_path[-1][1],
            'offline_error': 0,
            'distractor_path': distractor_path,
            'agent0': [_agent_x0,_agent_y0],
            'correct_side': correct_side,
            'path_prey_speed': self._prey_speed*_GAIN_PATH_PREY,
            'motion_steps_path_prey':0,
            'prey_distance_remaining_path_prey':prey_distance_remaining,
            'prey_distance_offset':_MAZE_Y - _AGENT_Y,
            'num_junctions': len(x_junction),
            'num_amb_junctions':np.int(sum([x > (1-self._path_prey_position) for x in x_junction])),
            'path_prey_position': self._path_prey_position,
            'n_correct_n_junction': n_correct_n_junction,
            'n_correct_n_amb_junction': n_correct_n_amb_junction,
            'n_trial': n_trial,
            'n_trial_amb': n_trial_amb,
            'x_exit': np.around(x_exit, decimals=3),
            'id_correct_offline': 0,
            'direction_fake_prey':  direction_fake_prey,
            'speed_fake_prey': speed_fake_prey,
            'distance_to_start': distance_to_start,
            'id_left0': -1,
            'reward': 1,
        }

        return state

    def create_agent(self, state):
        agent0= self._meta_state['agent0']
        agent = sprite.Sprite(
            x=agent0[0],  # agent_x0,  #  0.5,
            y=agent0[1], # _AGENT_Y,
            shape='square', # ,
            aspect_ratio=_AGENT_ASPECT_RATIO, # 4, # 3, #  1, 0.2,
            scale=_AGENT_SCALE,  # 0.1, # aspect_ratio=0.3, scale=0.05,
            mass = _DEFAULT_MASS,  # 1
            c0=64, c1=64, c2=64, # c0=128, c1=32, c2=32, # red
            metadata={'response_up': False, 'moved_h': False,'y_speed':0,'id_left0':-1},
        )
        if self._static_agent:
            agent.mass = np.inf

        state['agent'] = [agent]

    # def create_fake_prey(self, state):
    #
    #     distance_to_start = (_BALL_ON_DURATION/60) / (self._ms_per_unit/1000)   # 30/60 [s] / 2[s/unit] = 0.25
    #     direction_fake_prey = np.round(np.random.rand())*2-1
    #     speed = (1/60) / (self._ms_per_unit/1000)
    #
    #     fake_prey = sprite.Sprite(shape='circle', scale=_PREY_SCALE, c0=0, c1=255, c2=0,
    #                               metadata={'distance_to_start': distance_to_start, 'direction_fake_prey': direction_fake_prey,'speed':speed})
    #     fake_prey.position = [self._meta_state['prey_path'][0][0] +direction_fake_prey*distance_to_start,
    #                           self._meta_state['prey_path'][0][1]+self._prey_lead_in]
    #
    #     state['fake_prey'] = [fake_prey]

    def create_path_prey(self, state):
        path_prey = sprite.Sprite(shape='square', scale=0.04, 
        c0=_PATH_PREY_OPACITY, c1=_PATH_PREY_OPACITY, c2=_PATH_PREY_OPACITY)  # grey
        path_prey.position = [self._meta_state['prey_path'][0][0], self._meta_state['prey_path'][0][1]]

        if self._path_prey_opacity_staircase is None:
            self._path_prey_opacity = _PATH_PREY_OPACITY
        else:
            path_prey.opacity = self._path_prey_opacity_staircase.opacity

        state['path_prey'] = [path_prey]

    def meta_state_initializer(self):
        """Meta-state initializer."""
        return self._meta_state

################################################################################################
class Config():
    """Callable class returning config.
    
    All grid chase configs should inherit from this class.
    """

    def __init__(self,
                 stimulus_generator,
                 prey_opacity_staircase=None,
                 path_prey_opacity_staircase=None,
                 path_prey_position_staircase=None,
                 update_p_correct=None,
                 repeat_incorrect_trial=None,
                 fixation_phase=True,
                 prey_opacity=0,
                 path_prey_opacity=0,
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
        self._path_prey_opacity_staircase = path_prey_opacity_staircase
        self._path_prey_position_staircase = path_prey_position_staircase
        self._update_p_correct = update_p_correct
        self._repeat_incorrect_trial = repeat_incorrect_trial

        if self._prey_opacity_staircase is not None:
            self._prey_opacity = self._prey_opacity_staircase.opacity
        else:
            self._prey_opacity = prey_opacity

        if self._path_prey_opacity_staircase is not None:
            self._path_prey_opacity = self._path_prey_opacity_staircase.opacity
        else:
            self._path_prey_opacity = path_prey_opacity

        if self._path_prey_position_staircase is not None:
            self._path_prey_position = self._path_prey_position_staircase.path_prey_position
        else:
            self._path_prey_position = _PathPreyPosition_INIT_

        # How close to center joystick must be to count as joystick centering
        self._joystick_center_threshold = 0.05

        # Compute prey speed given ms_per_unit, assuming 60 fps
        self._prey_speed0 = 1000. / (60. * ms_per_unit) # 0.0083 frame width / refresh
        self._prey_speed = self._prey_speed0
        self._prey_lead_in = 0.15  # 0.08

        self._trial_init = TrialInitialization(
            stimulus_generator, prey_lead_in=self._prey_lead_in, prey_speed=self._prey_speed,
            static_prey=static_prey, static_agent=static_agent, prey_opacity_staircase=self._prey_opacity_staircase,
            path_prey_opacity_staircase=self._path_prey_opacity_staircase,path_prey_position_staircase=self._path_prey_position_staircase,
            update_p_correct=self._update_p_correct,repeat_incorrect_trial=self._repeat_incorrect_trial,ms_per_unit=ms_per_unit,
        )

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

        self._maze_walk_path = maze_lib.MazeWalk(
            speed=0., avatar_layer='path_prey', start_lead_in=self._prey_lead_in)

        self._fake_prey_walk = physics_custom.FakePreyWalk(
            speed=0,direction=np.array([0,0])
        )

        if self._static_prey:
            corrective_physics = []
        else:
            corrective_physics = [self._maze_walk, self._maze_walk_path,self._fake_prey_walk]
        
        self._physics = physics_lib.Physics(
            corrective_physics=corrective_physics)


    def _construct_task(self):
        """Construct task."""

        # prey_task = tasks_custom.TimeErrorReward(
        #      half_width=_TOOTH_HALF_WIDTH, # 40,  # given 60 Hz, 666*2/2 ms
        #      maximum=1,
        #      prey_speed=self._prey_speed,
        #      max_rewarding_dist = _MAX_REWARDING_DIST,
        #      prey_opacity_staircase = self._prey_opacity_staircase,
        # )
        #
        # # joystick_center_task = tasks_custom.BeginPhase('fixation')
        #
        # offline_task = tasks_custom.OfflineReward(
        #     'offMove',
        #     max_rewarding_dist=_MAX_REWARDING_DIST,
        #     path_prey_opacity_staircase=self._path_prey_opacity_staircase,
        #     path_prey_position_staircase=self._path_prey_position_staircase,
        #     update_p_correct=self._update_p_correct,
        #     repeat_incorrect_trial=self._repeat_incorrect_trial,
        #     )  # 0.1
        # # offline_timeout_task = tasks.Reset(
        # #     condition=lambda _, meta_state: meta_state['phase'] == 'motion_visible',
        # #     steps_after_condition=_REWARD,
        # # )

        online_task =  tasks_custom.ContactReward(
            reward_fn=1.,
            layers_0='agent',
            layers_1='prey',
            reset_steps_after_contact=np.inf,
            prey_opacity_staircase = self._prey_opacity_staircase, # TBD for online
            update_p_correct=self._update_p_correct,
            repeat_incorrect_trial=self._repeat_incorrect_trial,
        )

        timeout_task = tasks.Reset(
            condition=lambda _, meta_state: meta_state['phase'] == 'reward' ,
            # and meta_state['prey_distance_remaining']<0, # to prevent abort for H
            steps_after_condition=_REWARD,
        )
        self._task = tasks.CompositeTask(
            # # joystick_center_task,
            # offline_task,
            # # offline_timeout_task,
            timeout_task,
            online_task,
            # prey_task,
        )

    def _construct_action_space(self):
        """Construct action space."""
        self._action_space = action_spaces.Composite(
            eye=action_spaces.SetPosition(action_layers=('eye',), inertia=0.),
            hand=action_spaces_custom.JoystickColor(
                up_color=(128, 32, 32),  # red # (32, 128, 32), # green
                scaling_factor=_ACTION_SCALING_FACTOR),  # 0.01),
        )

    def _construct_game_rules(self):
        """Construct game rules."""

        def _make_transparent(s):
            s.opacity = 0

        def _set_prey_opacity(s):
            s.opacity = self._prey_opacity_staircase.opacity # self._prey_opacity

        def _set_path_prey_opacity(s):
            s.opacity = self._path_prey_opacity_staircase.opacity # self._prey_opacity

        def _make_opaque(s):
            s.opacity=255

        def _make_bright(s):
            s.c0 = 255
            s.c1 = 255
            s.c2 = 255

        def _make_green(s):
            s.c0 = 32
            s.c1 = 128
            s.c2 = 32

        def _make_red(s):
            s.c0 = 128
            s.c1 = 32
            s.c2 = 32

        ###########################################
        # 1. ITI phase (blank screen)
        ###########################################

        def _reset_physics(meta_state):
            self._maze_walk.set_prey_path(meta_state['prey_path'])
            self._maze_walk.speed = 0

            self._maze_walk_path.set_prey_path(meta_state['prey_path'])
            self._maze_walk_path.speed = 0

            self._fake_prey_walk.speed = 0
            self._fake_prey_walk.direction = np.array([0, 0])

        reset_physics = gr.ModifyMetaState(_reset_physics)

        phase_iti = gr.Phase(
            one_time_rules=reset_physics,
            duration=_ITI,  # 60 1sec
            name='iti',
        )
        ###########################################
        # 2. Joystick centering phase
        ###########################################
        appear_joystick = gr.ModifySprites(
            ['joystick_fixation', 'joystick'], _make_opaque)

        def _should_end_joystick_fixation(state):
            if state is not None:
                if state['joystick'] is not None:
                    joystick_pos = state['joystick'][0].position
                    dist_from_center = np.linalg.norm(joystick_pos - 0.5 * np.ones(2))
                    return dist_from_center < self._joystick_center_threshold
            else:
                return False

        phase_joystick_center = gr.Phase(
            one_time_rules=appear_joystick,
            end_condition=_should_end_joystick_fixation,
            name='joystick_fixation',
        )
        ###########################################
        # 3. Fixation phase (blank screen; 0 duration)
        ###########################################
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
            meta_state['fix_dur'] += 1
        increase_fixation_dur = gr.ConditionalRule(
            condition=_should_increase_fixation_dur,
            rules=gr.ModifyMetaState(_increase_fixation_dur)
        )
        reset_fixation_dur = gr.ConditionalRule(
            condition=lambda state, x: not _should_increase_fixation_dur(state, x),
            rules=gr.UpdateMetaStateValue('fix_dur', 0)
        )

        # end_condition
        def _should_end_fixation(state, meta_state):
            return (meta_state['fix_dur'] >= _FIXATION_STEPS) # now 0
        if not self._fixation_phase: # False in random_12_staircase_both_prey
            fixation_duration = 0
        else:
            fixation_duration = np.inf

        phase_fixation = gr.Phase(
            one_time_rules=[ disappear_joystick],  # appear_fixation
            continual_rules=[increase_fixation_dur, reset_fixation_dur],
            end_condition=_should_end_fixation,
            duration=fixation_duration,  # now 0
            name='fixation',
        )
        ###########################################
        # 4. ball on (500ms)
        ###########################################
        # one_time_rules
        disappear_fixation = gr.ModifySprites('fixation', _make_transparent)
        appear_fake_prey = gr.ModifySprites('fake_prey', _make_opaque)
        # create_fake_prey = custom_game_rules.CreateFakePrey(self._trial_init)

        def _unglue_fake_prey(meta_state):
            self._fake_prey_walk.speed = meta_state['speed_fake_prey']
            self._fake_prey_walk.direction = meta_state['direction_fake_prey']
        unglue_fake_prey = gr.ModifyMetaState(_unglue_fake_prey)

        # continual_rules
        def _update_motion_steps_fake_prey(meta_state):
            meta_state['distance_to_start'] -= meta_state['speed_fake_prey']
        update_motion_steps_fake_prey = gr.ModifyMetaState(_update_motion_steps_fake_prey)
        # def _update_motion_steps_fake_prey(s):
        #     s.position[0] = s.position[0]-s.metadata['direction_fake_prey']*s.metadata['speed']
        #     s.metadata['distance_to_start'] -= s.metadata['speed']
        # update_motion_steps_fake_prey = gr.ModifySprites('fake_prey', _update_motion_steps_fake_prey)

        # end condition
        def _end_ball_on_phase(state,meta_state):
            if meta_state['distance_to_start'] < 1e-4:
                return True
            return False

        phase_ball_on = gr.Phase(
            one_time_rules=[disappear_fixation, appear_fake_prey,unglue_fake_prey],  #  create_fake_prey
            continual_rules=[update_motion_steps_fake_prey],
            # duration=_BALL_ON_DURATION,  # 500 ms
            end_condition=_end_ball_on_phase,
            name='ball_on',
        )

        ###########################################
        # 5. maze on (500 ms)
        ###########################################
        # present maze
        glue_fake_prey = custom_game_rules.GlueFakePrey()
        disappear_fake_prey = gr.ModifySprites('fake_prey', _make_transparent)
        disappear_screen = gr.ModifySprites('screen', _make_transparent)

        # continual_rules
        def _increase_RT_offline(meta_state): # increase RT from when path prey is shown
            meta_state['RT_offline'] += 1
        increase_RT_offline = gr.ModifyMetaState(_increase_RT_offline)

        phase_maze_on = gr.Phase(
            one_time_rules=[glue_fake_prey,disappear_screen, disappear_fake_prey],
            continual_rules=[increase_RT_offline],
            duration=_MAZE_ON_DURATION,  # 500 ms
            name='maze_on',
        )

        ###########################################
        # 6. path_prey without agent (0)
        ###########################################
        # # one_time_rules
        # create_path_prey = custom_game_rules.CreatePathPrey(self._trial_init)
        # def _unglue_path_prey(meta_state):
        #     self._maze_walk_path.speed = self._prey_speed*_GAIN_PATH_PREY
        #     meta_state['path_prey_speed'] = self._prey_speed*_GAIN_PATH_PREY
        # unglue_path_prey = gr.ModifyMetaState(_unglue_path_prey)
        #
        # # continual_rules
        # def _update_motion_steps_path_prey(meta_state):
        #     meta_state['motion_steps_path_prey'] += 1 # [frames]? clock?
        #     meta_state['prey_distance_remaining_path_prey'] -= (self._prey_speed*_GAIN_PATH_PREY)
        # update_motion_steps_path_prey = gr.ModifyMetaState(_update_motion_steps_path_prey)
        # highlight_path = gr.ModifyOnContact('prey_wall','path_prey',modifier_0=_make_opaque)
        # def _decrease_path_prey_opacity(s,meta_state): #'path_prey_position'
        #     # if meta_state['prey_distance_remaining_path_prey'] < (meta_state['prey_distance_invisible']*_P_PATHPREY_DIM_DISTANCE): # P_DIM_DISTANCE=0 -> N/A
        #     if meta_state['prey_distance_remaining_path_prey'] < (
        #             meta_state['prey_distance_invisible'] * meta_state['path_prey_position']):  # P_DIM_DISTANCE=0 -> N/A
        #         s.opacity=0
        # dim_path_prey = custom_game_rules.DimPrey('path_prey',_decrease_path_prey_opacity)
        # glue_path_prey_conditional = gr.ConditionalRule(
        #     condition=lambda state, x: state['path_prey'][0].opacity == 0,
        #     rules=custom_game_rules.GluePathPrey()
        # )
        #
        # # end condition
        # def _end_path_prey_phase(state,meta_state):
        #     if meta_state['motion_steps_path_prey'] >= (meta_state['prey_distance_remaining'] / (self._prey_speed*_GAIN_PATH_PREY)): # [frames]? clock?
        #         return True
        #     return False
        #

        # phase_path_prey = gr.Phase(
        #     one_time_rules=[create_path_prey,unglue_path_prey], # disappear_screen,disappear_fake_prey,
        #     continual_rules=[update_motion_steps_path_prey,increase_RT_offline,dim_path_prey,glue_path_prey_conditional,highlight_path],  # highlight_path
        #     name='path_prey',
        #     duration=_PATH_PREY_DURATION, # 0
        #     end_condition=_end_path_prey_phase,
        # )
        ###########################################
        # 7. Offline movement phase ([paddleOn movementDone])
        ###########################################

        # one_time_rules
        create_agent = custom_game_rules.CreateAgent(self._trial_init)
        # glue_path_prey = custom_game_rules.GluePathPrey()
        set_path_prey_opacity = gr.ModifySprites('path_prey', _set_path_prey_opacity)  # self._prey_opacity

        # # continual_rules: booster away from exits
        # def _near_exit(state, meta_state):
        #     if len(state['agent']) > 0:
        #         agent = state['agent'][0]
        #         if meta_state['phase'] == 'offMove':
        #             distance_to_exits = agent.x - meta_state['x_exit']
        #             id_in_zone = np.any(np.abs(distance_to_exits)<_SLOWING_DIST)
        #         else:
        #             id_in_zone = False
        #     else:
        #         id_in_zone = False
        #     return id_in_zone
        # def _change_agent_mass(s):
        #     s.mass = _GAIN_MASS
        # def _change_agent_mass2(s):
        #     s.mass = _DEFAULT_MASS
        # update_agent_mass = gr.ConditionalRule(
        #     condition=lambda state, x: _near_exit(state, x),
        #     rules=gr.ModifySprites('agent', _change_agent_mass)
        # )
        # update_agent_mass2 = gr.ConditionalRule(
        #     condition=lambda state, x: not _near_exit(state, x),
        #     rules=gr.ModifySprites('agent', _change_agent_mass2)
        # )

        # continual_rules: change agent color if offline reward

        # def _reward(state, meta_state):
        #     if len(state['agent']) > 0:
        #         agent = state['agent'][0]
        #         if (meta_state['phase'] == 'offMove' and
        #                 agent.metadata['moved_h'] and
        #                 np.all(state['agent'][0].velocity == 0)): ##
        #             id_vertical=np.mod(meta_state['correct_side'],2)  # 0 for 0/2 (bottom/top)
        #             prey_exit_x = meta_state['prey_path'][-1][0]
        #             error_x = agent.x - prey_exit_x
        #             prey_exit_y = meta_state['prey_path'][-1][1]
        #             error_y = agent.y - prey_exit_y
        #             reward = max(0, 1 - np.abs((1-id_vertical)*error_x+id_vertical*error_y) / (_MAX_REWARDING_DIST + _EPSILON))
        #         else:
        #             reward = 0.
        #     else:
        #         reward = 0.
        #     return reward
        # def _offline_reward(state, meta_state):
        #     return _reward(state, meta_state) > 0
        # update_agent_color = gr.ConditionalRule(
        #     condition=lambda state, x: _offline_reward(state, x)>0,
        #     rules=gr.ModifySprites('agent', _make_green)
        # )
        # def _update_id_correct_offline(meta_state):
        #     meta_state['id_correct_offline'] = 1
        # update_id_correct_offline = gr.ConditionalRule(
        #     condition=lambda state, x: _offline_reward(state, x)>0,
        #     rules=gr.ModifyMetaState(_update_id_correct_offline)
        # )

        def _track_moved_h(s):
            if not np.all(s.velocity == 0): ##  # if not np.all(s.velocity[0] == 0): ##
                s.metadata['moved_h'] = True
                if s.velocity[0]<0:
                    s.metadata['id_left0'] = 1
                if s.velocity[0]>0:
                    s.metadata['id_left0'] = 0
        update_agent_metadata = gr.ModifySprites('agent', _track_moved_h)

        def _should_increase_RT_offline(state, meta_state):
            agent = state['agent'][0]
            return not agent.metadata['moved_h']
        update_RT_offline = gr.ConditionalRule(
            condition=_should_increase_RT_offline,
            rules=gr.ModifyMetaState(_increase_RT_offline)
        )

        # end_condition
        def _end_offline_phase(state,meta_state):
            agent = state['agent'][0]
            return agent.metadata['moved_h'] ## and np.all(agent.velocity == 0) and agent.metadata['y_speed'] == 0 ##
            # meta_state['joy_fix_post']>_joy_fix_post # np.all(agent.velocity == 0) #

        phase_off_move = gr.Phase(
            one_time_rules=[create_agent], # ,glue_path_prey], # ,set_path_prey_opacity],  # ,glue_path_prey],
            # [disappear_screen,disappear_fake_prey,create_agent,set_path_prey_opacity,glue_path_prey], # [disappear_fixation, disappear_screen, create_agent],
            continual_rules=[update_agent_metadata, update_RT_offline], # update_agent_mass,update_agent_mass2,update_id_correct_offline], # ,update_joystick_fixation_dur],  # update_agent_color
            name='offMove',
            end_condition=_end_offline_phase,  #  duration=10,
        )
        ###########################################
        # 8. Visible motion phase
        ###########################################
        # one_time_rules
        update_agent_color = gr.ModifySprites('agent', _make_bright)
        clear_prey_wall = gr.ModifySprites('prey_wall', _make_transparent)
        disappear_path_prey = gr.ModifySprites('path_prey', _make_transparent)
        def _unglue(meta_state): # slower prey
            # if meta_state['id_correct_offline'] != 1:
            #     self._prey_speed = self._prey_speed * _GAIN_SLOW_OFFLINE_ERROR
            self._maze_walk.speed = self._prey_speed
            meta_state['prey_speed'] = self._prey_speed
        unglue = gr.ModifyMetaState(_unglue)
        glue_agent = custom_game_rules.GlueAgent()
        make_agent_red = gr.ModifySprites('agent', _make_red)
        update_direction0_metastate=custom_game_rules.UpdateDirection0()

        def _should_update_id_correct(state, meta_state):
            if meta_state['end_x_agent'] < 0.5:  # target on left
                if meta_state['id_left0']==1:
                    return True
                if meta_state['id_left0']==0:
                    return False
            else:  # target on right
                if meta_state['id_left0']==0:
                    return True
                if meta_state['id_left0']==1:
                    return False
        def _should_not_update_id_correct(state, meta_state):
            return not _should_update_id_correct(state, meta_state)
        heavier_agent = gr.ConditionalRule(
            condition=_should_not_update_id_correct,
            rules=custom_game_rules.HeavierAgent()
        )
        red_agent = gr.ConditionalRule(
            condition=_should_not_update_id_correct,
            rules=gr.ModifySprites('agent', _make_red)
        )
        def _update_id_correct_offline(meta_state):
            meta_state['id_correct_offline'] = 1
        update_id_correct_offline = gr.ConditionalRule(
            condition=_should_update_id_correct,
            rules=gr.ModifyMetaState(_update_id_correct_offline)
        )

        # continual_rules
        def _update_motion_steps(meta_state):
            meta_state['motion_steps'] += 1 # [frames]? clock?
            meta_state['prey_distance_remaining'] -= self._prey_speed
        update_motion_steps = gr.ModifyMetaState(_update_motion_steps)

        # end_condition
        def _end_vis_motion_phase(state,meta_state):
            if meta_state['motion_steps'] > (self._prey_lead_in / self._prey_speed): # [frames]? clock?
                return True
            return False

        phase_motion_visible = gr.Phase(
            one_time_rules=[unglue,disappear_path_prey,update_agent_color,  # clear_prey_wall
                            update_direction0_metastate,heavier_agent,red_agent,update_id_correct_offline],  # make_agent_red glue_agent, unglue
            continual_rules=update_motion_steps,
            end_condition=_end_vis_motion_phase,  # duration=10,
            name='motion_visible',
        )
        ###########################################
        # 9. Invisible motion phase
        ###########################################
        # one_time_rules
        set_prey_opacity = gr.ModifySprites('prey', _set_prey_opacity)  # self._prey_opacity
        def _update_ts(meta_state):
            meta_state['ts'] = meta_state['prey_distance_remaining'] / self._prey_speed0 # [frames]
        update_ts = gr.ModifyMetaState(_update_ts)

        # continual_rules
        highlight_path = gr.ModifyOnContact('prey_wall','prey',modifier_0=_make_opaque)
        def _decrease_prey_opacity(s,meta_state):
            if meta_state['prey_distance_remaining'] < meta_state['prey_distance_invisible']*_P_DIM_DISTANCE: # P_DIM_DISTANCE=0 -> N/A
                s.opacity=0
        dim_prey = custom_game_rules.DimPrey('prey',_decrease_prey_opacity)
        def _increase_tp(meta_state):
            meta_state['tp'] += 1
        increase_tp = gr.ModifyMetaState(_increase_tp)

        def _compare_tp_ts(state,meta_state):
            prey_at_bottom = meta_state['tp']>=meta_state['ts']
            return prey_at_bottom
        glue_prey = gr.ConditionalRule(
            condition=_compare_tp_ts,
            rules=custom_game_rules.GluePrey()
        )

        smaller_prey=gr.ConditionalRule(
            condition=_compare_tp_ts,
            rules=custom_game_rules.SmallerPrey()
        )

        def _smaller_reward(meta_state):
            glue_time = _MAX_WAIT_TIME_GAIN * meta_state['ts']
            meta_state['reward'] /= meta_state['reward'] / glue_time
        smaller_reward=gr.ConditionalRule(
            condition=_compare_tp_ts,
            rules=gr.ModifyMetaState(_smaller_reward)
        )

        update_agent_color_green = gr.ModifyOnContact('agent','prey',modifier_0=_make_green)

        def _update_metadata(s):
            s.metadata['response_up']=True
        update_agent_metadata_online = gr.ModifyOnContact('agent', 'prey', modifier_0=_update_metadata)

        # end_condition
        def _end_motion_phase(state,meta_state):
            id_response_up = state['agent'][0].metadata['response_up']
            id_late = meta_state['tp'] > _MAX_WAIT_TIME_GAIN * meta_state['ts']
            # if meta_state['id_correct_offline'] == 1:  # response counts only if correct offline
            #     id_response_up = state['agent'][0].metadata['response_up']
            #     id_late = meta_state['tp'] > _MAX_WAIT_TIME_GAIN * meta_state['ts']
            # else:
            #     id_response_up = False
            #     id_late = meta_state['tp'] > (meta_state['prey_distance_invisible'] / self._prey_speed)
            return id_response_up or id_late

        phase_motion_invisible = gr.Phase(
            one_time_rules=[set_prey_opacity,update_ts],
            continual_rules=[update_motion_steps,increase_tp,glue_prey,update_agent_color_green,update_agent_metadata_online,
                             smaller_prey,smaller_reward,highlight_path],  # ,dim_prey], update_prey_distance
            end_condition=_end_motion_phase,
            name='motion_invisible',
        )
        ###########################################
        # 10. Reward Phase
        ###########################################
        # one_time_rules
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
        def _reset_prey_speed(meta_state):
            self._prey_speed = self._prey_speed0
        reset_prey_speed = gr.ModifyMetaState(_reset_prey_speed)

        phase_reward = gr.Phase(
            one_time_rules=[reveal_prey,reset_prey_speed], # make_agent_green,update_prey_color
            continual_rules=update_motion_steps,
            name='reward',
        )

        # Final rules
        phase_sequence = gr.PhaseSequence(
            phase_iti,  # 1sec
            phase_joystick_center, # variable
            phase_fixation, # 0
            phase_ball_on, # .5 sec
            phase_maze_on, # 0
            # phase_path_prey, # 0
            phase_off_move,  # variable [paddleOn moveDone]
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
