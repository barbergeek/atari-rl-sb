import time

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

#env = gym.make('ALE/Frogger-v5')
env = make_atari_env('ALE/Galaxian-v5', n_envs=1)
env = VecFrameStack(env, n_stack=4)


model = A2C.load("./Galaxian-v5-a2c-10000000.0-zoo-2023-04-25-16:36:33.333396/best_model")

obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    time.sleep(0.1)
    
exit()