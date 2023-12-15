import collections
import time
import random
import gym
from gym import spaces
from envs.metaenv import MetaWorldEnv


def initialize():
        # Register the environment
        gym.register(
            id='MetaWorld-v1',
            entry_point='envs.metaenv:MetaWorldEnv',
            kwargs={'params': None} 
        )