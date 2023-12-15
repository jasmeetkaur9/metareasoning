import gym
import sys
import  torch
from methods.ppo.ppo import PPOAgent
from omegaconf import OmegaConf 


if __name__ == '__main__':
    params = OmegaConf.load("/home/jk49379/metareasoning/config/params2.yml")
    env = gym.make(params.env)
    agent = PPOAgent(env, params)
    agent.train()

