from __future__ import division
import numpy as np
import time
import random
import operator
import mdptoolbox
from metaVI import MetaReasoningWorld
from metaEnv import MetaWorldEnv
from metaMCTS import MCTS
import metaNode as nd
from metaUtility import get_distributions
from metaUtility import get_execution_distributions
from metaUtility import random_policy
from metaUtility import round_robin_policy
import matplotlib.pyplot as plt
import scipy.stats as st

if __name__ == "__main__":
    samples = 1
    max_iter = 1000
    cost_values = np.zeros((max_iter + 2, samples + 1), dtype="float")
    ctime = np.zeros((max_iter + 2, samples + 1), dtype="float")
    for sample_num in range(0, samples):
        print("SAMPLE : ", sample_num+1)
        m = [5, 10, 5]
        v = [2, 1, 3]
        num_of_plans = 2
        actions_per_plan = 2
        max_planning_time = np.array([3, 3])
        deadline = 5 # 6
        actions = [1, 2]
        dist, planning_times = get_distributions(num_of_plans, actions_per_plan, max_planning_time, m, v)
        e_dist, e_times = get_execution_distributions(num_of_plans, actions_per_plan, max_execution_time=3)
        # print(dist)
        # print(e_dist)
        env = MetaWorldEnv(num_of_plans, actions_per_plan, deadline, actions, max_planning_time, False)
                           #dist,planning_times,e_dist,e_times)
        mw = MetaReasoningWorld(env,False)

        # Print the number of successful terminal states, total terminal states, total states
        # print(env.successStates())
        # Print the distribution
        # env.print_for_me()

        print("DO VALUE ITERATION")
        # Make a call to the metaVI
        # Call to run VI
        st_time = time.time()
        its, v, p, t = mw.do_value_iteration(100)
        t1 = time.time() - st_time

        # Or Use the toolbox
        # Initialize the arguments as per requirement
        P = env.transition_model
        P = np.swapaxes(P, 0, 1)
        reward_model = np.zeros((env.num_of_states, env.num_of_plans), dtype="float")
        for i in range(0, env.num_of_states):
            for j in range(0, env.num_of_plans):
                reward_model[i][j] = env.reward_model[i]
        R = reward_model
        vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)

        # Call to run VI
        st_time = time.time()
        vi.run()
        t1 = time.time() - st_time
        p = list(vi.policy)
        p = [x + 1 for x in p]

        # Calculate the Expected Reward
        total_reward = 0.0
        runs = 1000
        s = []
        for i in range(0, runs):
            pp, state_path, reward = mw.get_policy_from_path(p)
            total_reward = total_reward + reward
            s.append(reward)
        total_reward = total_reward / runs
        print(np.mean(s))
        # print(np.std(s))
        # print(st.t.interval(confidence=0.95, df=len(s) - 1, loc=np.mean(s), scale=st.sem(s)))

        print("DO ROUND ROBIN")
        # Do Round Robin
        for time_steps in range(1,deadline+1):
            p = round_robin_policy(time_steps, 1, actions, 100)
            # Calculate the Expected Reward
            total_reward = 0.0
            runs = 1000
            s = []
            for i in range(0, runs):
                state_path, reward = mw.get_solution_using_policy(p)
                total_reward = total_reward + reward
                s.append(reward)
            total_reward = total_reward / runs
            print(np.mean(s))

        print("DO RANDOM")
        # Do Random
        p = random_policy(actions, 100)
        # Calculate the Expected Reward
        total_reward = 0.0
        runs = 1000
        s = []
        for i in range(0, runs):
            state_path, reward = mw.get_solution_using_policy(p)
            total_reward = total_reward + reward
            s.append(reward)
        total_reward = total_reward / runs
        print(np.mean(s))

        print("DO MCTS")
        list_k = []
        list_k_sd = []

        # k is the number of rollout iterations
        for k in [8000]:
            num_of_trials = 1000
            total_reward = 0.0
            s = []
            for i in range(0, num_of_trials):
                st_time = time.time()
                curr_id = 0
                pp = []
                reward = 0.0
                state_path = []
                while True:
                    reward = reward + env.reward_model[curr_id]
                    root_node = nd.Node(curr_id)
                    mcts = MCTS(root_node, env, False)
                    best_action = mcts.Run(k)
                    pp.append(best_action)
                    state_path.append(env.get_state_from_id(curr_id))
                    res, _, _ = env.step(curr_id, best_action)
                    list1 = list(res.keys())
                    list2 = list(res.values())
                    curr_id = (random.choices(list1, weights=list2, k=1))[0]
                    if env.done(curr_id):
                        reward = reward + env.reward_model[curr_id]
                        break
                pp.pop()
                total_reward = total_reward + reward
                s.append(reward)
            list_k.append(np.mean(s))
            list_k_sd.append(np.std(s))
            t1 = time.time() - st_time
            # print(st.t.interval(confidence=0.95, df=len(s) - 1, loc=np.mean(s), scale=st.sem(s)))
            # ctime[k][sample_num] = t1
            # cost_values[k][sample_num] = np.mean(s)
        for i in range(0, len(list_k)):
            print(list_k[i])
        # for i in range(0, len(list_k)):
        #     print(list_k_sd[i])
