import numpy as np




def random_agent(env, episodes, agent_type):
    reward_random = []
    for i in range(int(episodes)):
        episode_reward = []
        if agent_type == 1:
            curr_state, info = env.reset()
            curr_id = env.observation_space.get_state_id(curr_state)
            curr = curr_id
        else :
            curr, info = env.reset()
        sum_reward = 0 
        while True:
            action = env.action_space.sample()
            next_state, next_id, reward, _, done, info  = env.step_mw(action, curr_state)
            curr_state = next_state
            curr_id = next_id
            sum_reward = sum_reward + reward
            if done:
                break
        episode_reward.append(sum_reward)
        reward_random.append(np.mean(np.array(episode_reward)))
    return reward_random

def rr_agent(env, episodes, agent_type, step = 1):
    reward_rr = []
    action_n = np.arange(env.action_space.n)
    n = env.action_space.n
    for i in range(int(episodes)):
        episode_reward = []
        if agent_type == 1:
            curr_state, info = env.reset()
            curr_id = env.observation_space.get_state_id(curr_state)
            curr = curr_id
        else :
            curr, info = env.reset()
        sum_reward = 0 
        a = 0
        while True:
            action = action_n[a%n]
            a+=1
            next_state, next_id, reward, _, done, info  = env.step_mw(action, curr_state)
            curr_state = next_state
            curr_id = next_id
            sum_reward = sum_reward + reward
            if done:
                break
        episode_reward.append(sum_reward)
        reward_rr.append(np.mean(np.array(episode_reward)))
    return reward_rr

    