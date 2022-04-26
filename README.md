# ACER and DQN agent for atariAgents
Just click download zip files to get all necessary files
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pre-requisite packages. Install two sets of packages in two different virtual environment

To install pre-requisite packages for ACER_trainer

```bash
pip install -r requirementsforACER.txt
```

To install pre-requisite packages for DQN_trainer

```bash
pip install -r requirementsforDQN.txt
```

## Usage
DQN_trainer

assumed pre-requisite packages have been installed in an new virtual environment
```python
uncomment first two methods of bottom three methods to train and test the agent on space invader games
then
python DQN_trainer.py

or 

uncomment the line of save method in train_agent() method, change its saving path to you need, after training is completed saved weights will be stored in given path, use loading_trainedAgent() + testing_agent() to check results

python DQN_trainer.py
```

ACER_trainer.py

assumed pre-requisite packages have been installed in an new virtual environment
```bash
python ACER_trainer.py
```
```python
in terminal, an user interface will pop up
under main menu, type train or evaluate to select wanted function of the program

then second menu pop up, 
select wanted game for training by type its game number(1 for space invader; 2 for assult; 3 for lunar lander)
```
## saved weights
```python
There are too many weights that cannot be submitted to github due to their filesize

So i select weights of each major training time steps like 10k, 1m, 100m etc. for each games

by defaut, filepath used in evaluate_best() method in ACER_trainer.py is the best weights for each relevant game.
```