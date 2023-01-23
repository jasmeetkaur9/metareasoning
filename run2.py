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
    x = np.random.normal(loc=mean, scale=variance, size=1000)
    index = random.randint(2, max_planning_time_)
    count, bins_count = np.histogram(x, index + 1)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    dist1 = np.zeros(max_planning_time_ + 10)
    index_i = 0
    for i in range(0, index):
        dist1[i] = cdf[i]
    for i in range(index, max_planning_time_ + 10):
        dist1[i] = 1.0
    for i in range(0, max_planning_time_+10):
        if dist1[i] >= 1.0:
            index_i = i
            break
    return dist1, index_i


def get_distributions(num_of_plans, num_of_actions, max_planning_time, mean, variance):
    dist = np.zeros((num_of_plans, num_of_actions, max_planning_time + 10), dtype="float")
    planning_times = np.zeros((num_of_plans, num_of_actions), dtype="int")
    for i in range(0, num_of_plans):
        for j in range(0, num_of_actions):
            arr, index = get_single_distribution(max_planning_time, mean[i], variance[i])
            dist[i][j] = arr
            planning_times[i][j] = index

    return dist, planning_times

def get_execution_distributions(num_of_plans, num_of_actions,max_execution_time):
    mean = 20
    scale = 5
    e_dist = np.zeros((num_of_plans, num_of_actions, max_execution_time), dtype="float")
    e_planning_times = np.zeros((num_of_plans, num_of_actions), dtype="int")
    for i in range(0, num_of_plans):
        for j in range(0, num_of_actions):
            data = np.random.normal(loc=mean, scale=scale,size=max_execution_time)
            s = np.array(data)
            s = s/s.sum()
            s = np.sort(s)
            e_dist[i][j] = s
            e_planning_times[i][j] = max_execution_time-1

    return e_dist,e_planning_times

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
        samples = 1
        max_iter = 10
        cost_values = np.zeros((max_iter+2,samples+1), dtype="float")
        ctime = np.zeros((max_iter+2,samples+1), dtype="float")
        for sample_num in range(0, samples):
            m = [5, 10, 5]
            v = [2, 1, 3]
            num_of_plans = 2
            actions_per_plan = 2
            max_planning_time = 5
            # total_time = (2 * 3 * max_planning_time) + 2  #remove total_time from the formulation
            deadline = 6
            actions = [1, 2]
            dist, planning_times = get_distributions(num_of_plans, actions_per_plan, max_planning_time, m, v)
            e_dist, e_times = get_execution_distributions(num_of_plans,actions_per_plan,max_execution_time=3)
            env = MetaWorldEnv(num_of_plans, actions_per_plan, deadline, actions, max_planning_time)
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
            for i in range(0, 10):
                pp, state_path, solution_cost = mw.get_policy_from_path(p)
                print(pp)
                print(solution_cost)
                print(state_path)
                cost1 = cost1 + solution_cost
            cost1 = cost1/10
            ctime[0][sample_num] = t1
            cost_values[0][sample_num] = cost1
            print(cost1)

            #Do MCTS
            for k in [10,100]:
                print("k is : ",k)
                runs = 10
                for i in range(0, runs):
                    st_time = time.time()
                    curr_id = 0
                    cost = 0.0
                    pp = []
                    state_path = []
                    while True:
                        cost = cost + env.reward_model[curr_id]
                        n = nd.Node(curr_id)
                        mcts = MCTS(n, env, False)
                        best_action = mcts.Run(k)
                        pp.append(best_action)
                        state_path.append(env.get_state_from_id(curr_id))
                        res, _, _ = env.step(curr_id, best_action)
                        if (env.done(curr_id)):
                            break
                        list1 = list(res.keys())
                        list2 = list(res.values())
                        curr_id = (random.choices(list1, weights=list2, k=1))[0]
                    print(cost)
                    pp.pop()
                    print(pp)
                    print(state_path)
                t1 = time.time() - st_time
                ctime[k][sample_num] = t1
                cost_values[k][sample_num] = cost
        avg = []
        std = []
        # for k in range(0, max_iter+1):
        #     avg.append(np.mean(cost_values[k]))
        #     print(avg[k])
        # for k in range(0, max_iter+1):
        #     std.append(np.std(cost_values[k]))
        #     print(std[k])


