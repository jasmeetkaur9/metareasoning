import gym
import sys
import  torch
import argparse
from methods.ppo.ppo import PPOAgent
from omegaconf import OmegaConf 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()
    config = args.config_file

    params = OmegaConf.load(config)
    env = gym.make(params.env)
    agent = PPOAgent(env, params)
    agent.train()

