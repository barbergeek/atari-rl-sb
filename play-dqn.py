import time
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env

env = make_atari_env('ALE/Frogger-v5', n_envs=4)

model = DQN.load("./Frogger-v5-dqn-10000000.0-zoo-2023-04-21-16:12:33.261800 - best/best_model")

obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    time.sleep(0.1)
