import os
import time
import random
import argparse

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from omegaconf import OmegaConf 

from methods.mcts.mcts import MCTSAgent
from utils.utils import initialize




if __name__ == '__main__':

    params = OmegaConf.load("/home/jk49379/metareasoning/config/metaenv.yml")
    env = gym.make(params.env)
    _, _ = env.reset(seed = int(params.seed))

    if params.verbose == "True":
        while True:
            action = 0
            next_state, reward, _, done, info  = env.step_mw(action, curr_state)
            print(curr_state," : ", env.get_state_from_id(curr_state), next_state, " : ", env.get_state_from_id(next_state), reward, done)
            curr_state = next_state
            if done:
                break

    reward = 0
    done = False
    agent = MCTSAgent(env, params)
    episodes = int(params.episodes)
    max_episode_steps = int(params.max_episode_steps)
    timestr = time.strftime("%Y-%m-%d")
    avg_reward = 0


    for i in range(int(params.episodes)):
        ob = env.reset()
        env._max_episode_steps = max_episode_steps
        video_path = os.path.join(
            params.video_basepath, f"output_{timestr}_{i}.mp4")
        rec = VideoRecorder(env, path=video_path)
        sum_reward = 0
        node = None
        all_nodes = []
        cp = int(params.cp)
        
        
        while True:
            path.append(curr_state)
            action, node, cp = agent.run(params, node)
            if params.env == "MetaWorld-v1":
                next_state, reward, _, done, info  = env.step_mw(action, curr_state)
                curr_state = next_state
            else :
                next_state, reward, _, done, info  = env.step(action)
            sum_reward = sum_reward + reward
            if done:
                break
        avg_reward = avg_reward + sum_reward
        
        
        if params.verbose == "True":
            print("Solution Path Found", sum_reward)
            for i in range(0, len(path)):
                print(env.get_state_from_id(path[i]))

    trials = int(params.episodes)
    avg_reward = avg_reward/trials
    print(avg_reward)

        
            
