import os
import time
import random
import argparse
import numpy as np

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from omegaconf import OmegaConf 

from methods.mcts.mcts import MCTSAgent, Node
from envs.metaenv import MetaWorldEnv
from utils.utils import initialize, plot_graph



if __name__ == '__main__':
    initialize()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()
    config = args.config_file

    # Analysis for deadline

    deadline = np.arange(0, 50)
    params = OmegaConf.load(config)

    deadline_data = []
    for i in range(len(deadline)):
        params['deadline'] = str(deadline[i])
        env = gym.make(params.env, params = params)
        curr_state, _ = env.reset()
        if params.verbose == "True":
            while True:
                action = 0
                if params.env == "MetaWorld-v1":
                    next_state, reward, _, done, info = env.step_mw(action, curr_state)
                    print(curr_state, reward, next_state)
                    curr_state = next_state
                else :
                    next_state, reward, _ , done, infor = env.step(action)
                    print(next_state, reward)
                if done :
                    break
        reward = 0
        done = False
        agent = MCTSAgent(env, params)
        episodes = int(params.episodes)
        max_episode_steps = int(params.max_episode_steps)
        timestr = time.strftime("%Y-%m-%d")
        avg_reward = []
        episode_reward = []

        for i in range(int(params.episodes)):
            curr_state, _ = env.reset()
            env._max_episode_steps = max_episode_steps
            sum_reward = 0
            node = Node(False, 0, curr_state)
            all_nodes = []
            cp = int(params.cp)
            path = []
            while True:
                action, node, cp = agent.run(params, node)
                if params.env == "MetaWorld-v1":
                    next_state, reward, _, done, info = env.step_mw(action, curr_state)
                    curr_state = next_state
                else :
                    next_state, reward, _, done, info = env.step(action)
                sum_reward = sum_reward + reward
                if done :
                    break
            episode_reward.append(sum_reward)
        deadline_data.append(np.mean(np.array(episode_reward)))
    plot_graph(deadline_data, "Deadline", "deadline_score" )
