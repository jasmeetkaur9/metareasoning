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
import matplotlib.pyplot as plt
import scipy.stats as st

if __name__ == "__main__":
    samples = 1
    max_iter = 1000
    cost_values = np.zeros((max_iter + 2, samples + 1), dtype="float")
    ctime = np.zeros((max_iter + 2, samples + 1), dtype="float")
    for sample_num in range(0, samples):
        print("SAMPLE : ", sample_num)
        m = [5, 10, 5]
        v = [2, 1, 3]
        num_of_plans = 2
        actions_per_plan = 2
        max_planning_time = np.array([3, 3])
        deadline = 6
        actions = [1, 2]
        dist, planning_times = get_distributions(num_of_plans, actions_per_plan, max_planning_time, m, v)
        e_dist, e_times = get_execution_distributions(num_of_plans, actions_per_plan, max_execution_time=3)
        # print(dist)
        # print(e_dist)
        env = MetaWorldEnv(num_of_plans, actions_per_plan, deadline, actions, max_planning_time, False)
        mw = MetaReasoningWorld(env,False)

        # Print the number of successful terminal states, total terminal states, total states
        # print(env.successStates())
        # Print the distribution
        env.print_for_me()

        print("DO Value Iteration")
        # You can make a call to the metaVI
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
        runs = 100
        s = []
        for i in range(0, runs):
            pp, state_path, reward = mw.get_policy_from_path(p)
            total_reward = total_reward + reward
            s.append(reward)
        total_reward = total_reward / runs
        print(np.mean(s))
        print(np.std(s))
        # print(st.t.interval(confidence=0.95, df=len(s) - 1, loc=np.mean(s), scale=st.sem(s)))

        print("DO MCTS")
        list_k = []
        list_k_sd = []

        # k is the number of rollout iterations
        for k in [1000]:
            num_of_trials = 10
            total_reward = 0.0
            s = []
            for i in range(0, num_of_trials):
                st_time = time.time()
                curr_state = 0
                pp = []
                reward = 0
                state_path = []
                test = 0
                # Till the end of episode
                while True:
                    reward = reward + env.reward_model[curr_state]
                    root_node = nd.Node(curr_state)

                    mcts = MCTS(root_node, env, False)
                    best_action = mcts.Run(k)

                    pp.append(best_action)
                    state_path.append(env.get_state_from_id(curr_state))
                    next_state, _ = env.step_next_state(curr_state, best_action)
                    curr_state = next_state
                    if env.done(curr_state):
                        reward = reward + env.reward_model[curr_state]
                        break
                    test = test + 1
                    pp.pop()
                    total_reward = total_reward + reward
                    s.append(reward)
                print(np.mean(s))
                list_k.append(np.mean(s))
                list_k_sd.append(np.std(s))
                t1 = time.time() - st_time
                # print(st.t.interval(confidence=0.95, df=len(s) - 1, loc=np.mean(s), scale=st.sem(s)))
            # for i in range(0, len(list_k)):
            #     print(list_k[i])
            # for i in range(0, len(list_k)):
            #     print(list_k_sd[i])
        avg = []
        moving_avg = []
        std = []
