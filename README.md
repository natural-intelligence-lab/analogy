# MWorks codebase

## 4 tasks: analogy, memory-guided saccade, ready-set-go, pursuit

1. Install MWorks. Preferred version is the "bleeding edge"
   nightly build of MWorks, which you can install by downloading the "Nighly
   Build" on the [MWorks downloads page](https://mworks.github.io/downloads/). 

2. Open the following mwel files for each task in MWorks client:
   - analogy: analogy_debug.mwel
   - memory-guided saccade: MemorySaccadeNumber_debug.mwel
   - ready-set-go: RSG_twoPrior_handEye_debug.xml
   - pursuit: SmoothPursuit_record.mwel
Note that each mwel file has a corresponding version whose name does not have "_debug".
That version was actually used in the experiment as it was set up with LabJack and eye trackers. 
Note that ready-set-go task has an older version of stimulus code (xml) and pursuit task doesn't 
have the "debug" version (i.e., you would get an error when loading the task).

3. For RSG, you can do the online performance monitoring by loading matlab_stairBallAlphaProd.m
in the matlab interface.
