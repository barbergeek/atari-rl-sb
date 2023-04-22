import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure

env = make_atari_env('ALE/Frogger-v5', n_envs=1)
model = PPO.load("best_model-ppo")

obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    time.sleep(0.1)
    
exit()