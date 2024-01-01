import os
import time
import random
import argparse

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from omegaconf import OmegaConf 
import numpy as np
import envs
from envs.pr2 import PR2

params = OmegaConf.load("/home/jk49379/metareasoning/config/pr2.yml")
p = int(params.num_symbolic_plans)
c = int(params.num_actions_per_plan)
pr2 = PR2(p,c)
env = gym.make(params.env, params = params)
env.set_pybullet_ob(pr2)
curr_state, _ = env.reset()


reward = 0
done = False
episodes = int(params.episodes)
max_episode_steps = int(params.max_episode_steps)
timestr = time.strftime("%Y-%m-%d")
avg_reward = 0
agent_type = 1

reward_random = []
for i in range(int(episodes)):
    episode_reward = []
    if agent_type == 1:
        print("reset")
        curr_state, info = env.reset()
        curr_id = env.observation_space.get_state_id(curr_state)
        curr = curr_id
    else :
        curr, info = env.reset()
    sum_reward = 0
    c = 0 
    while True:
        action = 1
        next_state, next_id, reward, _, done, info  = env.step_mw(action, curr_state)
        curr_state = next_state
        curr_id = next_id
        sum_reward = sum_reward + reward
        if done:
            break
    episode_reward.append(sum_reward)
    reward_random.append(np.mean(np.array(episode_reward)))


print(reward_random)
