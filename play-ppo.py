import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure

env = make_atari_env('ALE/Frogger-v5', n_envs=4)
model = PPO.load("./Frogger-v5-ppo-10000000.0-2023-04-21-08:23:52.560473 - best/best_model")

obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    time.sleep(0.06)
    
exit()