from gym import spaces
import gym

# Register the environment
gym.register(
    id='MetaWorld-v1',
    entry_point='envs.metaenv:MetaWorldEnv',
    kwargs={'params': None} 
)


gym.register(
    id='PR2-v0',
    entry_point='envs.pybulletenv:PR2Env',
    kwargs={'params': None} 
)