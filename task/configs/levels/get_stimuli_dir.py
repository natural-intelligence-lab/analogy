"""Get stimuli directory."""

import os


def stimuli_dir():

    if not 'MWorks' in os.getcwd():
        # Running the python demo, so give relative path
        return '../stimuli'
    else:
        # # Laptop hansem
        # return (
        #     '/Users/hansem/Documents/MazeSetGo/task/stimuli'
        # )

        # Psychophysics rig
        return '/Users/jazlab/Documents/MazeSetGo/task/stimuli'

        # Laptop nwatters
        # return (
        #     '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/'
        #     'maze_set_go/task/stimuli'
        # )
