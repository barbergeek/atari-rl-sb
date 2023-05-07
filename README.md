# atari-rl-sb

Atari Reinforcement Learning with Stable Baselines 3

Virginia Tech Master of Information Techonolgy  
Spring 2023 AI Innovation & Machine Learning (ECE 5494)  
Scott Hoge

## Installation
Conda environment in environment.yml

`$ conda env create -n <environment name>    # default atari-rl-sb`

### Training modules for Atari 2600 Frogger
`$ python train-<model>.py               # for default hyperparameters`  
`$ python train-<model>-zoo.py           # for RL zoo recommended hyperparameters`

Model is a2c, dqn, or ppo

Best training model saved to `./<game>-<model>-<timesteps>[-zoo]-<date>/best_model.zip`

Example: `./Frogger-v5-ppo-10000000.0-2023-04-21-08:23:52.560473/best_model.zip`

### Play existing models
`$ python play-[a2c,dqn,ppo].py          # plays and shows the best model for the specified algorithm`

## Other games
`$ python train-a2c-zoo-[cc,gal,mspacman].py # trains CrazyClimber, Galaxian, MsPacman`  
`$ python play-a2c-[cc,gal,mspacman].py      # plays and shows CrazyClimber, Galaxian, MsPacman`  
