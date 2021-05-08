"""Functions to show trial videos and images."""

import sys
sys.path.append('../../task')

import json
from matplotlib import pyplot as plt
from moog import sprite as sprite_lib
import numpy as np

from utils import common


def get_frames(trial_path,
               sample_every,
               translucent_prey=False,
               include_iti=False,
               include_fixation=True,
               include_delay=True,
               include_motion=True):
    """Get list of frames."""
    trial = json.load(open(trial_path, 'r'))
    step_indices = np.arange(0, len(trial) - 2, sample_every)
    
    observer = common.observer()
    
    state = common.get_initial_state(trial)
    frames = []
    eye_pos = []
    motion_started = False
    for step in step_indices:
        step_string = trial[step + 2]
        common.update_state(state, step_string, rules=[])
        
        phase = step_string[4][1]['phase']
        if not include_iti and phase == 'iti':
            continue
        if not include_fixation and phase == 'fixation':
            continue
        if not include_delay and phase == 'delay':
            continue
        if not include_motion and phase == 'motion':
            continue

        if phase == 'motion_invisible':
            for s in state['prey']:
                if translucent_prey:
                    s.opacity = 128
                else:
                    s.opacity = 0
        
        frames.append(observer(state))
        eye_pos.append(state['eye'][0].position)

    return frames, eye_pos


def display_video(frames, eye_pos=None):
    ##  Display video of trial

    # Need small figure size or else will get cut off in Jupyter for some weird
    # reason
    fig, ax = plt.subplots(1, 1, figsize=(2.3, 2.3))
    ax.set_xticks([])
    ax.set_yticks([])
    imshow = ax.imshow(frames[0], extent=[0, 1, 1, 0])
    scatter = ax.scatter([], [], marker='.', color='r', s=20)
    display_frames = []
    for i, frame in enumerate(frames):
        imshow.set_data(frame)
        if eye_pos is not None:
            if i > 0:
                scatter.set_offsets(
                    np.array([0, 1]) + np.array([1, -1]) * eye_pos[0:i])
                color = np.clip(np.arange(1 - 0.01 * i, 1, 0.01), 0., 1.)
                color = np.stack(
                    (color, 0.3 * np.ones_like(color), 1. - color), axis=1)
                scatter.set_color(color)
        fig.canvas.draw()
        if eye_pos is not None:
            display_frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
        else:
            display_frames.append(frame)
        plt.pause(0.01)

    return display_frames
