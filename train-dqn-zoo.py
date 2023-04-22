import os
import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack

from callbacks import DumpBestModelInfo

startTime = datetime.datetime.now()

# initialization parameters
game = "Frogger-v5"     # should be part of the Atari Learning Environment space (ALE/{game})
num_environments = 4    # number of training environments
model_type = "dqn"      # model to use (change the model in import and code if you change this)
seed = 12345            # rng seed
wrapper_kwargs= dict(frame_skip=0)

# atari:
#   env_wrapper:
#     - stable_baselines3.common.atari_wrappers.AtariWrapper
#   frame_stack: 4
#   policy: 'CnnPolicy'
#   n_timesteps: !!float 1e7
#   buffer_size: 100000
#   learning_rate: !!float 1e-4
#   batch_size: 32
#   learning_starts: 100000
#   target_update_interval: 1000
#   train_freq: 4
#   gradient_steps: 1
#   exploration_fraction: 0.1
#   exploration_final_eps: 0.01
#   # If True, you need to deactivate handle_timeout_termination
#   # in the replay_buffer_kwargs
#   optimize_memory_usage: False

frame_stack = 4
n_timesteps = 1e7
learning_rate = 1e-4
buffer_size = 100000
batch_size = 32
learning_starts = 100000
target_update_interval = 1000
train_freq = 4
gradient_steps = 1
exploration_fraction = 0.1
exploration_final_eps = 0.01
optimize_memory_usage = False

# where model and tensorboard files will go
# make a new directory for each run, identifying game, model, and run size (steps)
# example: Frogger-v5-dqn-16-2023-04-19-22:00:00.00000
save_path = os.path.join(".",f"{game}-{model_type}-{n_timesteps}-zoo-{str(startTime).replace(' ','-')}")

# Initialize game environment
env = make_atari_env(f"ALE/{game}", n_envs=num_environments, wrapper_kwargs=wrapper_kwargs)
env = VecFrameStack(env, n_stack=frame_stack)
env.reset()

# Instantiate the model
model = DQN("CnnPolicy", env, verbose=0, learning_rate=learning_rate, learning_starts=learning_starts, 
            buffer_size=buffer_size, batch_size=batch_size, target_update_interval=target_update_interval, train_freq=train_freq,
            gradient_steps=gradient_steps, exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps,
            optimize_memory_usage=optimize_memory_usage)
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