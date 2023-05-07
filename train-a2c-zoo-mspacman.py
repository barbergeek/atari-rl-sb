import os
import datetime

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from callbacks import DumpBestModelInfo

startTime = datetime.datetime.now()

# initialization parameters
game = "MsPacman-v5"     # should be part of the Atari Learning Environment space (ALE/{game})
model_type = "a2c"      # model to use (change the model in import and code if you change this)
seed = 12345            # rng seed
wrapper_kwargs= dict(frame_skip=0)

# Hyperparameters from rl-baselines3-zoo
"""
atari:
  env_wrapper:
    - stable_baselines3.common.atari_wrappers.AtariWrapper
  # Equivalent to
  # vec_env_wrapper:
  #   - stable_baselines3.common.vec_env.VecFrameStack:
  #         n_stack: 4
  frame_stack: 4
  policy: 'CnnPolicy'
  n_envs: 16
  n_timesteps: !!float 1e7
  ent_coef: 0.01
  vf_coef: 0.25
  policy_kwargs: "dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))"
"""
frame_stack = 4     # default for Atari wrapping
n_envs = 16         # number of training environments
n_timesteps = 1e8   # x10 steps
ent_coef = 0.01     #
vf_coef = 0.25      #
policy_kwargs = dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))

# where model and tensorboard files will go
# make a new directory for each run, identifying game, model, and run size (steps)
# example: Frogger-v5-dqn-16-2023-04-19-22:00:00.00000
save_path = os.path.join(".",f"{game}-{model_type}-{n_timesteps}-zoo-{str(startTime).replace(' ','-')}")

# Initialize game environment
env = make_atari_env(f"ALE/{game}", n_envs=n_envs, wrapper_kwargs=wrapper_kwargs)
env = VecFrameStack(env, n_stack=frame_stack)
env.reset()

model = A2C("CnnPolicy", env, verbose=0, policy_kwargs=policy_kwargs, ent_coef=ent_coef, vf_coef=vf_coef)
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