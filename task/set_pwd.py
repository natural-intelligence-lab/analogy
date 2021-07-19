"""Set pwd variable depending on laptop vs rig."""

import os

# # # OLD Psychophysics rig
# _PWD = '/Users/jazlab/Documents/MazeSetGo/task'
# _PYTHON_SITE_PACKAGES = (
#     '/Users/jazlab/miniconda/envs/nwatters/lib/python3.8/site-packages'
# )

# Local nwatters
# _PWD = '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/maze_set_go/task'
# _PYTHON_SITE_PACKAGES = (
#     '/Users/nicholaswatters/miniconda3/envs/mworks_moog/lib/python3.8/site-packages'
# )

if getvar('platform') == 'laptop':
    _PWD = '/Users/hansem/Documents/MazeSetGo/task'
    _PYTHON_SITE_PACKAGES = (
        '/opt/anaconda3/envs/mworks/lib/python3.8/site-packages'
    )
elif getvar('platform') == 'desktop':
    _PWD = '/Users/hansem/Documents/MazeSetGo/task'
    _PYTHON_SITE_PACKAGES = (
        '/Users/hansem/miniconda3/envs/mworks/lib/python3.8/site-packages'
    )
elif getvar('platform') == 'psychophysics':
    # laptop hansem
    _PWD = '/Users/hansem/Documents/MazeSetGo/task'
    _PYTHON_SITE_PACKAGES = (
        '/opt/anaconda3/envs/mworks/lib/python3.8/site-packages'
    )
elif getvar('platform') == 'monkey':
    # laptop hansem
    _PWD = '/Users/hansem/Documents/MazeSetGo/task'
    _PYTHON_SITE_PACKAGES = (
        '/opt/miniconda3/envs/mworks/lib/python3.9/site-packages'
    )
else:
    raise ValueError('Invalid platform')

# Update pwd and python_site_packages variables in mworks
setvar('pwd', _PWD)
setvar('python_site_packages', _PYTHON_SITE_PACKAGES)