"""Set pwd variable depending on laptop vs rig."""

import os

# Rig
_PWD = '/Users/jazlab/Documents/nwatters/maze_pong/task'
_PYTHON_SITE_PACKAGES = (
    '/Users/jazlab/miniconda/envs/nwatters/lib/python3.8/site-packages'
)

# Local nwatters
# _PWD = '/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/maze_pong/task'
# _PYTHON_SITE_PACKAGES = (
#     '/Users/nicholaswatters/miniconda3/envs/mworks_moog/lib/python3.8/'
#     'site-packages'
# )

# Update pwd and python_site_packages variables in mworks
setvar('pwd', _PWD)
setvar('python_site_packages', _PYTHON_SITE_PACKAGES)