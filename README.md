# Maze-Set-Go

## Getting Started

This directory only serves as an example of how to run MOOG from MWorks.

To run this example, follow these steps:

1. Install MWorks. Currently (as of 03/18/2021), you need the "bleeding edge"
   nightly build of MWorks, which you can install by downloading the "Nighly
   Build" on the [MWorks downloads page](https://mworks.github.io/downloads/). 
2. Create a virtual environment with python version 3.8. If you are using conda,
   this can be done with `conda create -n your_env_name python=3.8`
3. Activate your newly created virtual environment and install MOOG with `pip install moog-games`.
4. Navigate to where you want this code to live on your computer and clone this repo with `git clone https://github.mit.edu/jazlab/MazeSetGo.git`.
5. Check that the python task runs by navigating to `task/python_demo` and running `$ python3 run_demo.py` --- you should be able to play this with your keyboard.
6. Before running MWorks, edit the paths in `task/set_pwd.py` to refer to your current working directory and the python site packages for the virtual environment you created above.
7. Also edit `task/configs/levels/get_stimuli_dir.py` similarly.
8. Now you can launch mworks and run `tasks/moog.mwel`. You might want to edit the eye and controller interfaces at the top of that mwel file.
