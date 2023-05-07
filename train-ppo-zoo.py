import os
import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack

from callbacks import DumpBestModelInfo

startTime = datetime.datetime.now()

# initialization parameters
game = "Frogger-v5"     # should be part of the Atari Learning Environment space (ALE/{game})
model_type = "ppo"      # model to use (change the model in import and code if you change this)
seed = 12345            # rng seed
wrapper_kwargs= dict(frame_skip=0)

# zoo hyperparameters
# atari:
#   env_wrapper:
#     - stable_baselines3.common.atari_wrappers.AtariWrapper
#   frame_stack: 4
#   policy: 'CnnPolicy'
#   n_envs: 8
#   n_steps: 128
#   n_epochs: 4
#   batch_size: 256
#   n_timesteps: !!float 1e7
#   learning_rate: lin_2.5e-4
#   clip_range: lin_0.1
#   vf_coef: 0.5
#   ent_coef: 0.01

frame_stack = 4
n_envs = 8    # number of training environments
n_steps = 128
n_epochs = 4
batch_size = 256
n_timesteps = 1e7
learning_rate = 2.5e-4
clip_range = 0.1
vf_coef = 0.5
ent_coef = 0.01

# where model and tensorboard files will go
# make a new directory for each run, identifying game, model, and run size (steps)
# example: Frogger-v5-dqn-16-2023-04-19-22:00:00.00000
save_path = os.path.join(".",f"{game}-{model_type}-{n_timesteps}-zoo-{str(startTime).replace(' ','-')}")

# Initialize game environment
env = make_atari_env(f"ALE/{game}", n_envs=n_envs, wrapper_kwargs=wrapper_kwargs)
env = VecFrameStack(env, n_stack=frame_stack)
env.reset()

# Instantiate the model
model = PPO("CnnPolicy", env, verbose=0, n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size,
            learning_rate=learning_rate, clip_range=clip_range, vf_coef=vf_coef, ent_coef=ent_coef)
model.set_random_seed(seed)

#set up logger
new_logger = configure(save_path, ["csv","tensorboard"])
model.set_logger(new_logger)

#custom callback to dump best model data
dump_data = DumpBestModelInfo(model_type)

# Use EvalCallback to evaluate model
eval_callback = EvalCallback(env, best_model_save_path=save_path, log_path=save_path,
                             eval_freq = 10000, deterministic=True, render=False, callback_on_new_best=dump_data)

# train
model.learn(total_timesteps=n_timesteps, progress_bar=True, callback=eval_callback)

# save the training results
model.save(os.path.join(save_path,f"{game}-{model_type}-{n_timesteps}"))

# report run (clock) time
delta = datetime.datetime.now() - startTime
print("Elapsed time = {}".format(delta))

exit()