from stable_baselines import ACER
from stable_baselines.common import callbacks
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.callbacks import BaseCallback
import numpy as np
import gym
import os
from os import walk
# LunarLander-v2 train_lunar log_lunar train train_Assault log_Assault 
CHECKPOINT_DIR_assult='./train_assult/'
CHECKPOINT_DIR_lunar='./train_lunar/'
CHECKPOINT_DIR_space='./train_space/'
LOG_DIR='./log_Assault/'
LOG_DIR_Lunar='./log_lunar/'

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
            model_path = os.path.join(self.save_path, 'best_model_phrase3_{}'.format(self.n_calls))
            self.model.save(model_path)
        
        return True

def train_lunar():
    environment_name = 'LunarLander-v2'
    callback= SaveOnBestTrainingRewardCallback(check_freq=10000,save_path=CHECKPOINT_DIR_lunar)
    env = gym.make(environment_name)
    env = DummyVecEnv([lambda: env])
    model = ACER('MlpPolicy', env, verbose = 1, tensorboard_log=LOG_DIR_Lunar)
    model.learn(total_timesteps=100000, callback=callback)

# LunarLander-v2 SpaceInvadersNoFrameskip-v4 AssaultNoFrameskip-v0
def train_model(game=None):
    if game is not None:
        if game == 1:
            environment_name = 'SpaceInvadersNoFrameskip-v4'
            callback= SaveOnBestTrainingRewardCallback(check_freq=10000,save_path=CHECKPOINT_DIR_space)
            env = make_atari_env(environment_name, num_env=1, seed=0)
            env = VecFrameStack(env, n_stack=4)# running multiple copies of same environement
        if game == 2:
            environment_name = 'AssaultNoFrameskip-v0'
            callback= SaveOnBestTrainingRewardCallback(check_freq=10000,save_path=CHECKPOINT_DIR_assult)
            env = make_atari_env(environment_name, num_env=1, seed=0)
            env = VecFrameStack(env, n_stack=4)# running multiple copies of same environement
    else:
        environment_name = 'SpaceInvadersNoFrameskip-v4'
        callback= SaveOnBestTrainingRewardCallback(check_freq=10000,save_path=CHECKPOINT_DIR_space)
        env = make_atari_env(environment_name, num_env=1, seed=0)
        env = VecFrameStack(env, n_stack=4)# running multiple copies of same environement
    # env = gym.make(environment_name)
    # env = DummyVecEnv([lambda: env])
    # model = ACER('MlpPolicy', env, verbose = 1)

    """
    The ACER (Actor-Critic with Experience Replay) model class, https://arxiv.org/abs/1611.01224

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) The discount value
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param num_procs: (int) The number of threads for TensorFlow operations

        .. deprecated:: 2.9.0
            Use `n_cpu_tf_sess` instead.

    :param q_coef: (float) The weight for the loss on the Q value
    :param ent_coef: (float) The weight for the entropy loss
    :param max_grad_norm: (float) The clipping value for the maximum gradient
    :param learning_rate: (float) The initial learning rate for the RMS prop optimizer
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                              'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param rprop_epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
        (default: 1e-5)
    :param rprop_alpha: (float) RMSProp decay parameter (default: 0.99)
    :param buffer_size: (int) The buffer size in number of steps
    :param replay_ratio: (float) The number of replay learning per on policy learning on average,
                         using a poisson distribution
    :param replay_start: (int) The minimum number of steps in the buffer, before learning replay
    :param correction_term: (float) Importance weight clipping factor (default: 10)
    :param trust_region: (bool) Whether or not algorithms estimates the gradient KL divergence
        between the old and updated policy and uses it to determine step size  (default: True)
    :param alpha: (float) The decay rate for the Exponential moving average of the parameters
    :param delta: (float) max KL divergence between the old policy and updated policy (default: 1)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    model = ACER('CnnPolicy', env, verbose = 1, tensorboard_log=LOG_DIR, gamma=0.99, n_steps=20, num_procs=None, q_coef=0.5, ent_coef=0.01, max_grad_norm=10,
                 learning_rate=7e-4, lr_schedule='linear', rprop_alpha=0.99, rprop_epsilon=1e-5, buffer_size=5000,
                 replay_ratio=4, replay_start=1000, correction_term=10.0, trust_region=True,
                 alpha=0.99, delta=1,_init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)
    # model= ACER.load("./train/best_model_phrase2_33590000",env=env, tensorboard_log=LOG_DIR)
    model.learn(total_timesteps=3000000, callback=callback)
    print("Finished Trainging")

def evaluate_best(game=1):
    # env = make_atari_env('SpaceInvadersNoFrameskip-v4', num_env=1, seed=0)
    # env = VecFrameStack(env, n_stack=4)
    if game == 1:
        env = make_atari_env('SpaceInvadersNoFrameskip-v4', num_env=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
        model= ACER.load("./train_space/best_model_phrase2_31710000",env=env)
        meanreward,std = evaluate_policy(model, env, n_eval_episodes=10, render=True)
        print('Score:{}'.format(meanreward))
    if game == 2:
        env = make_atari_env('AssaultNoFrameskip-v0', num_env=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
        model= ACER.load("./train_assult/best_model_phrase2_1450000",env=env)
        meanreward,std = evaluate_policy(model, env, n_eval_episodes=10, render=True)
        print('Score:{}'.format(meanreward))
    if game == 3:
        environment_name = 'LunarLander-v2'
        env = gym.make(environment_name)
        env = DummyVecEnv([lambda: env])
        model= ACER.load("./train_lunar/best_model_phrase2_90000",env=env)
        meanreward,std = evaluate_policy(model, env, n_eval_episodes=10, render=True)
        print('Score:{}'.format(meanreward))
    # environment_name = 'LunarLander-v2'
    # env = gym.make(environment_name)
    # env = DummyVecEnv([lambda: env])
    # model= ACER.load("./train_lunar/best_model_phrase3_50000.zip",env=env)
    # # meanreward,std = evaluate_policy(model, env, n_eval_episodes=3, render=True)

    # obs = env.reset()
    # episodes=10
    # for episode in range(1, episodes+1):
    #     done = False
    #     score=0

    #     while not done:
    #         action,_states = model.predict(obs)
    #         # print(action)
    #         obs,reward,done,info = env.step(action)
    #         env.render()
    #         score+=reward
    #     print('Episode:{} Score:{}'.format(episodes,score[0]))
    # env.close()

def evaluate_all(filename,game=1):
    if game == '1':
        env = make_atari_env('SpaceInvadersNoFrameskip-v4', num_env=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
        model= ACER.load("./train_space/"+filename,env=env)
    if game == '2':
        env = make_atari_env('AssaultNoFrameskip-v0', num_env=1, seed=0)
        env = VecFrameStack(env, n_stack=4)
        model= ACER.load("./train_assult/"+filename,env=env)
    if game == '3':
        environment_name = 'LunarLander-v2'
        env = gym.make(environment_name)
        env = DummyVecEnv([lambda: env])
        model= ACER.load("./train_lunar/"+filename,env=env)

    meanreward,std = evaluate_policy(model, env, n_eval_episodes=1, render=True)
    return meanreward

state_flag=0
def menu():
    print("\ntype relevant commands to use:")
    print("train")
    print("evaluate")
    print("find optimal result(developer used only)")
    print("type 0 to exit")
    print(":",end="")

def menu2():
    print("\ntype number of relevant commands:")
    print("1. space invaders")
    print("2. assult")
    print("3. lunar lander")
    print("type 9 to return to previous page")
    print(":",end="")

def result_selector(select=3):
    f = []
    results=[]
    if (select=="1"):
        filename="train_space"
        for (dirpath, dirnames, filenames) in walk(filename):
            f.extend(filenames)
            break

        for item in f:
            results.append(evaluate_all(item, game=select))
        print(max(results))
        print(results)
        return max(results)

    if (select=="2"):
        filename="train_assult"
        for (dirpath, dirnames, filenames) in walk(filename):
            f.extend(filenames)
            break

        for item in f:
            results.append(evaluate_all(item, game=select))
        print(max(results))
        print(results)
        return max(results)

    if (select=="3"):
        filename="train_lunar"
        for (dirpath, dirnames, filenames) in walk(filename):
            f.extend(filenames)
            break

        for item in f:
            results.append(evaluate_all(item, game=select))
        print(max(results))
        print(results)
        return max(results)

while(True):
    if state_flag==0:
        menu()
    if state_flag==1:
        menu2()
    if state_flag==2:
        menu2()

    x = input()
    commands=x.split(' ')
    if(commands[0] == '0'):
        exit()
    if(commands[0] == 'train'):
        state_flag=1
    if(commands[0] == 'evaluate'):
        state_flag=2
    if(commands[0] == 'find'):
        result_selector(commands[1])
        continue
    if(state_flag ==1):
        if(commands[0] == '9'):
            state_flag=0
        if(commands[0] == '1'):
            train_model(1)
        if(commands[0] == '2'):
            train_model(2)
        if(commands[0] == '3'):
            train_lunar()
    if(state_flag ==2):
        if(commands[0] == '9'):
            state_flag=0
        if(commands[0] == '1'):
            evaluate_best(1)
        if(commands[0] == '2'):
            evaluate_best(2)
        if(commands[0] == '3'):
            evaluate_best(3)
    else:
        print("Please type correct command!\n")

# train_model(2)
# train_lunar()
# evaluate_model()