# PofLS
---

### Dependencies
 - opencv
 - numpy
 - matplotlib

### Usage
#### Run an experiment
- Execute the `main.py` file with no command line arguments to create a GUI for the experiment
    - The program will prompt you to enter a string as the experiment id in the command line 
    - Follow the instructions on the GUI to run an experiment with arbitrarily many rounds
    - The proportion of three kinds of rounds are 1:1:1 (Go:Stop:Decide), which can be adjust in `main.py` line 26
    - The experiment result `{experiment_id}.json` is saved to `./data/` upon pressing `Q` to quit the experiment

#### Plot the data
- Execute the `main.py` file with paths to all the json files to be plotted
    - e.g. `python main.py ./data/qyd.json ./data/haha.json`




