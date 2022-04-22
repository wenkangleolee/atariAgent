from stable_baselines import ACER
from stable_baselines.common import callbacks
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.callbacks import BaseCallback
import numpy as np
import os


CHECKPOINT_DIR='./train_Assault/'
LOG_DIR='./log_Assault/'

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self,check_freq: int,save_path: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback,self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def _init_callback(self) -> None:
        if self.save_path is not None:
            return super()._init_callback()
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq ==0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        
        return True

def train_model():
    environment_name = 'AssaultNoFrameskip-v0'
    callback= SaveOnBestTrainingRewardCallback(check_freq=10000,save_path=CHECKPOINT_DIR)
    env = make_atari_env(environment_name, num_env=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    model = ACER('CnnPolicy', env, verbose = 1, tensorboard_log=LOG_DIR)
    # model= ACER.load("./train/best_model_4800000",env=env)
    model.learn(total_timesteps=10000, callback=callback)
    print("Finished Trainging")

#AssaultNoFrameskip-v0 SpaceInvadersNoFrameskip-v4
def evaluate_model():
    env = make_atari_env('SpaceInvadersNoFrameskip-v4', num_env=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    model= ACER.load("./train/best_model_phrase2_33450000",env=env)
    # env = make_atari_env('AssaultNoFrameskip-v0', num_env=1, seed=0)
    # env = VecFrameStack(env, n_stack=4)
    # model= ACER.load("./train_Assault/best_model_phrase2_1200000",env=env)
    s, t=evaluate_policy(model, env, n_eval_episodes=20, return_episode_rewards=True, render=True)
    for i in s:    
        print('Score:{}'.format(i))

    obs = env.reset()
    episodes=1
    for episode in range(1, episodes+1):
        done = False
        score=0

        while not done:
            action,_states = model.predict(obs)
            # print(action)
            from time import sleep
            sleep(0.0416)
            obs,reward,done,info = env.step(action)
            env.render()
            score+=reward
        print('Episode:{} Score:{}'.format(episode,score))
    env.close()

evaluate_model()
# train_model()