import time

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env('ALE/MsPacman-v5', n_envs=1)
env = VecFrameStack(env, n_stack=4)


model = A2C.load("./MsPacman-v5-a2c-100000000.0-zoo-2023-04-26-18:04:11.548049/best_model")

obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    time.sleep(0.1)
    
exit()