import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from metrics import MetricLogger
from agent import AtariAgent
from wrappers import ResizeObservation, SkipFrame

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

startTime = datetime.datetime.now()

# Initialize game environment
env = make_atari_env('ALE/Frogger-v5', n_envs=4)

env.reset()

model = A2C("CnnPolicy", env, verbose=0)
#model = PPO.load("ppo-frogger", env)

#set up logger
tmp_path = "./sb3_log-a2c/"
#new_logger = configure(tmp_path, ["stdout","csv","tensorboard"])
new_logger = configure(tmp_path, ["csv","tensorboard"])
model.set_logger(new_logger)

# Use EvalCallback to evaluate model
eval_callback = EvalCallback(env, best_model_save_path="./a2c-model/", log_path=tmp_path,
                             eval_freq = 10000, deterministic=True, render=False)

#eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)

# train
#model.learn(total_timesteps=env.num_envs * 1_000_000, progress_bar=True)
#model.learn(total_timesteps=4_000_000, progress_bar=True, reset_num_timesteps=False)
model.learn(total_timesteps=4_000_000, progress_bar=True, callback=eval_callback)

# save the training results
model.save("a2c-frogger")

exit()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # Path('frogger_100k_episodes.chkpt')
game = AtariAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

#episodes = 100000
episodes = 100

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state, info = env.reset()

    # Play the game!
    while True:

        # 4. Run agent on the state
        action = game.act(state)

        # 5. Agent performs action
        next_state, reward, truncated, terminated, info = env.step(action)
        done = truncated or terminated

        # 6. Remember
        game.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = game.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done:
            break

    logger.log_episode()

    if e % 50 == 0:
        logger.record(
            episode=e,
            epsilon=game.exploration_rate,
            step=game.curr_step
        )

# save the final model
game.save()

delta = datetime.datetime.now() - startTime
print("Elapsed time = {}".format(delta))