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
from utils.utils import plot_graph
import envs
from methods.baselines.baseline import random_agent, rr_agent


# TODO : Need to make it compatible with both envs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()
    config = args.config_file

    params = OmegaConf.load(config)
    env = gym.make(params.env, params = params)
    curr_state, _ = env.reset()
    with open((os.path.join(params.logdir, 'exp_7_config.txt')), 'w') as f:
        config_str = OmegaConf.to_yaml(params)
        f.write(config_str)

    if params.verbose == "True":
        while True:
            action = 0
            next_state, next_id, reward, _, done, info  = env.step_mw(action, curr_state)
            print(curr_state," : ", curr_state, next_state, " : ", next_state, reward, done)
            curr_state = next_state
            if done:
                break

    reward = 0
    done = False
    agent = MCTSAgent(env, params)
    episodes = int(params.episodes)
    max_episode_steps = int(params.max_episode_steps)
    timestr = time.strftime("%Y-%m-%d")
    avg_reward = []
    episode_reward = []
    if params.env == "MetaWorld-v1":
        agent_type = 1
    else :
        agent_type = 0

    for i in range(int(params.episodes)):
        if agent_type == 1:
            curr_state, info = env.reset()
            curr_id = env.observation_space.get_state_id(curr_state)
            curr = curr_id
        else :
            curr, info = env.reset()
        env._max_episode_steps = max_episode_steps
        sum_reward = 0
        node = Node(False, 0, curr)
        all_nodes = []
        cp = int(params.cp)
        path = []
        while True:
            path.append(curr_state)
            action, node, cp = agent.run(params, node)
            if agent_type == 1:
                next_state, next_id, reward, _, done, info  = env.step_mw(action, curr_state)
                curr_state = next_state
                curr_id = next_id
            else :
                next_state, reward, _, done, info  = env.step(action)
            sum_reward = sum_reward + reward
            if done:
                break
        episode_reward.append(sum_reward)
        avg_reward.append(np.mean(np.array(episode_reward)))
        if params.verbose == "True":
            print("Solution Path Found", sum_reward)
            for i in range(0, len(path)):
                print(path[i])

    env = gym.make(params.env, params = params)
    random_result = random_agent(env, episodes, agent_type)
    env = gym.make(params.env, params = params)
    rr_result = rr_agent(env, episodes, agent_type, 1)

    # Plot results
    labels = ["Random", "Round Robin", "MCTS"]
    data = []
    data.append(random_result)
    data.append(rr_result)
    data.append(avg_reward)
    plot_graph(data, labels, "Episodes", "Reward", "exp_7", 1)

        