from __future__ import division
from copy import deepcopy
from mcts import mcts
from functools import reduce
import numpy as np
from itertools import count
from collections import defaultdict
import collections
import time
import random
import operator
import mdptoolbox
from metaEnvTest import MetaWorldEnvM
from metaPlan import MetaReasoningWorld
from metaEnv import MetaWorldEnv
from metaMcts import Action,State

MAX_STATES = 10000

#tm = np.zeros((MAX_STATES, 2, MAX_STATES), dtype="float")


def get_single_distribution(max_planning_time_, mean, variance):
    random_index = np.random.randint(1, max_planning_time_)
    x = np.random.normal(loc=mean, scale=variance, size=1000)
    count, bins_count = np.histogram(x, bins=random_index + 1)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    dist1 = np.zeros(max_planning_time_)
    index = 0
    for i in range(0, random_index):
        dist1[i] = cdf[i]
    for i in range(random_index, max_planning_time_):
        dist1[i] = 1.0
    for i in range(0, max_planning_time_):
        if dist1[i] >= 1.0:
            index = i
            break
    return dist1, index


def get_distributions(num_of_plans, num_of_actions, max_planning_time, mean, variance):
    dist = np.zeros((num_of_plans, num_of_actions, max_planning_time), dtype="float")
    planning_times = np.zeros((num_of_plans, num_of_actions), dtype="int")
    for i in range(0, num_of_plans):
        for j in range(0, num_of_actions):
            arr, index = get_single_distribution(max_planning_time, mean[i], variance[i])
            dist[i][j] = arr
            planning_times[i][j] = index

    return dist, planning_times


def test():
    env_ = MetaWorldEnvM()
    mw_ = MetaReasoningWorld(env_)
    its_, v_, p_, t_ = mw_.do_value_iteration(100)
    print("Size of State Space ", env_.num_of_states)
    print("Computation Time in secs ", t_)
    print("Resultant policy", mw_.get_policy_from_path(p_))


if __name__ == "__main__":

    # test()
    for t_ in range(11, 12):
        ctime1 = 0.0
        ctime2 = 0.0
        space_size = 0
        samples = 10
        for i in range(0, samples):
            m = [5, 10]
            v = [2, 1]
            num_of_plans = 2
            actions_per_plan = 3
            max_planning_time = t_
            # total_time = (2 * 3 * max_planning_time) + 2  #remove total_time from the formulation
            deadline = t_ - 3
            actions = [1, 2]
            dist, planning_times = get_distributions(num_of_plans, actions_per_plan, max_planning_time, m, v)
            # DEFAULT_DIST2 = [[[0.05, 0.462, 0.943, 1., 1., 1.],
            #                   [0.179, 0.863, 1., 1., 1., 1.],
            #                   [0.044, 0.44, 0.917, 1., 1., 1.]],
            #
            #                  [[0.018, 0.149, 0.526, 0.86, 0.992, 1.],
            #                   [0.022, 0.235, 0.731, 0.968, 1., 1.],
            #                   [0.012, 0.197, 0.683, 0.977, 1., 1.]]]
            # DEFAULT_TIMES2 = np.array([[3, 2, 3], [5, 4, 4]])
            # dist = DEFAULT_DIST2
            # planning_times = DEFAULT_TIMES2
            # print(dist)
            # print(planning_times)
            # global tm
            # tm = np.zeros((MAX_STATES, 2, MAX_STATES), dtype="float")
            env = MetaWorldEnv(num_of_plans, actions_per_plan, deadline, actions, max_planning_time, dist,
                               planning_times)

            mw = MetaReasoningWorld(env)

            #its, v, p, t = mw.do_value_iteration(100)

            P = env.transition_model
            P = np.swapaxes(P, 0, 1)
            reward_model = np.zeros((env.num_of_states, env.num_of_plans), dtype="float")
            for i in range(0, env.num_of_states):
                for j in range(0, env.num_of_plans):
                    reward_model[i][j] = env.reward_model[i]
            R = reward_model
            vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)

            st_time = time.time()
            vi.run()
            t1 = time.time() - st_time

            p = list(vi.policy)
            p = [x + 1 for x in p]

            # print("Size of State Space ", env.num_of_states)
            # print("Computation Time in secs ", t)
            # print("Resultant policy", mw.get_policy_from_path(p))
            ctime1 = ctime1 + t1

            # DO MCTS

            start_state_id = env.get_id_from_state(env.start_state)
            initial_state = State(env, start_state_id)
            curr_state = initial_state
            p = []
            st_time = time.time()
            for i in range(0, env.deadline):
                from mcts import mcts
                mcts = mcts(iterationLimit=100)
                bestAction = mcts.search(initialState=curr_state)
                p.append(bestAction)
                next_state = initial_state.takeAction(bestAction)
            t2 = time.time() - st_time
            ctime2 = ctime2 + t2
            space_size = space_size + env.num_of_states

        print("Avg State Size", space_size / samples)
        print("Avg Time with Value Iteration", ctime1 / samples)
        print("Avg Time with MCTS", ctime2 / samples)
