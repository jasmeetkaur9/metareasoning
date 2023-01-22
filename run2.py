from __future__ import division
import numpy as np
import time
import random
import operator
import mdptoolbox
from metaPlan import MetaReasoningWorld
from metaEnv import MetaWorldEnv
from metaEnvTest import MetaWorldEnvM
from single_player import MCTS
import Node as nd
import matplotlib.pyplot as plt

random.seed(40)
MAX_STATES = 20000


# tm = np.zeros((MAX_STATES, 2, MAX_STATES), dtype="float")


def get_single_distribution(max_planning_time_, mean, variance):
    random_index = np.random.randint(1, max_planning_time_)
    x = np.random.normal(loc=mean, scale=variance, size=1000)
    count, bins_count = np.histogram(x, bins=random_index + 1)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    dist1 = np.zeros(max_planning_time_ + 10)
    index = 0
    for i in range(0, random_index):
        dist1[i] = cdf[i]
    for i in range(random_index, max_planning_time_ + 10):
        dist1[i] = 1.0
    for i in range(0, max_planning_time_):
        if dist1[i] >= 1.0:
            index = i
            break
    return dist1, index


def get_distributions(num_of_plans, num_of_actions, max_planning_time, mean, variance):
    dist = np.zeros((num_of_plans, num_of_actions, max_planning_time + 10), dtype="float")
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
    for t_ in range(4, 5):
        space_size = 0
        samples = 100
        cost_values = np.zeros((1002), dtype="float")
        ctime = np.zeros((1002), dtype="float")
        for i in range(0, samples):
            m = [5, 10, 5]
            v = [2, 1, 3]
            num_of_plans = 2
            actions_per_plan = 2
            max_planning_time = 6
            # total_time = (2 * 3 * max_planning_time) + 2  #remove total_time from the formulation
            deadline = 7
            actions = [1, 2]
            dist, planning_times = get_distributions(num_of_plans, actions_per_plan, max_planning_time, m, v)
            # print(dist)
            # print(planning_times)
            env = MetaWorldEnv(num_of_plans, actions_per_plan, deadline, actions, max_planning_time, dist, planning_times)
            mw = MetaReasoningWorld(env)

            # its, v, p, t = mw.do_value_iteration(100)

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
            cost1 = 0.0
            for ss in range(0, 10):
                pp, _, solution_cost = mw.get_policy_from_path(p)
                cost1 = cost1 + solution_cost
            cost1 = cost1 / 10
            ctime[0] = t1
            cost_values[0] += cost1

            #Do MCTS
            l = []
            l.append(cost1)
            moving_average = []
            moving_average.append(np.mean(l))
            for k in range (1, 1000):
                cost = 0.0
                st_time = time.time()
                for i in range(1,10):
                    n = nd.Node(0)
                    mcts = MCTS(n,env,False)
                    mcts.Run(k)
                    cost = cost + n.sputc
                t1 = time.time() - st_time
                t1 = t1/10
                ctime[k] = t1
                l.append(cost/10)
                cost_values[k] += (cost/10)
                moving_average.append(np.mean(l))
            # plt.plot(l)
            # plt.show()

        cost_values = cost_values/samples
        #print(list(cost_values))
        print(cost_values[0],cost_values[10], cost_values[50], cost_values[100], cost_values[500], cost_values[1000])
        plt.plot(list(cost_values))
        plt.show()
        print("Value Iteration : ", cost_values[0], ctime[0])
        print("State Space ", env.num_of_states)

