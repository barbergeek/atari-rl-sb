import os
import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback

from callbacks import DumpBestModelInfo

startTime = datetime.datetime.now()

# initialization parameters
game = "Frogger-v5"     # should be part of the Atari Learning Environment space (ALE/{game})
num_environments = 4    # number of training environments
steps = 16              # in millions
model_type = "ppo"      # model to use (change the model in import and code if you change this)
seed = 12345            # rng seed
wrapper_kwargs= dict(frame_skip=0)  # Frogger-v5 already frameskips 4, so don't do it again

# where model and tensorboard files will go
# make a new directory for each run, identifying game, model, and run size (steps)
# example: Frogger-v5-dqn-16-2023-04-19-22:00:00.00000
save_path = os.path.join(".",f"{game}-{model_type}-{steps}-{str(startTime).replace(' ','-')}")

# Initialize game environment
env = make_atari_env(f"ALE/{game}", n_envs=num_environments, wrapper_kwargs=wrapper_kwargs)
env.reset()

# Instantiate the model
model = PPO("CnnPolicy", env, verbose=0)
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
model.learn(total_timesteps=steps * 1_000_000, progress_bar=True, callback=eval_callback)

# save the training results
model.save(os.path.join(save_path,f"{game}-{model_type}-{steps}"))

# report run (clock) time
delta = datetime.datetime.now() - startTime
print("Elapsed time = {}".format(delta))

exit()