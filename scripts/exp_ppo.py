import gym
import sys
import  torch
import argparse
from methods.ppo.ppo import PPOAgent
from omegaconf import OmegaConf
import envs 
from envs.pr2 import PR2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()
    config = args.config_file

    params = OmegaConf.load(config)
    if params.env == "MetaWorld-v1" or params.env ==  "PR2-v0":
        agent_type = 1
    else :
        agent_type = 0
    if agent_type == 1:
        env = gym.make(params.env, params = params)
        if params.env ==  "PR2-v0":
            p = int(params.num_symbolic_plans)
            c = int(params.num_actions_per_plan)
            pr2 = PR2(p,c)
            env.set_pybullet_ob(pr2)
        curr_state, _ = env.reset()
    else :
        env = gym.make(params.env)
    agent = PPOAgent(env, params)
    agent.train()

