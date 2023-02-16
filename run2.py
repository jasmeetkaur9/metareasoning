from __future__ import division
import numpy as np
import time
import random
import operator
import mdptoolbox
from metaPlan import MetaReasoningWorld
from metaEnv import MetaWorldEnv
from single_player import MCTS
import Node as nd
import matplotlib.pyplot as plt
import scipy.stats as st

random.seed(10)
MAX_STATES = 20000


# tm = np.zeros((MAX_STATES, 2, MAX_STATES), dtype="float")


def get_single_distribution(max_planning_time_, mean, variance,padding,mpt):
    x = np.random.normal(loc=mean, scale=variance, size=1000)
    index = random.randint(1, max_planning_time_)
    count, bins_count = np.histogram(x, index + 1)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    dist1 = np.zeros(mpt+padding)
    index_i = 0
    for i in range(0, index):
        dist1[i] = cdf[i]
    for i in range(index, mpt+padding):
        dist1[i] = 1.0
    for i in range(0, mpt+padding):
        if dist1[i] >= 1.0:
            index_i = i
            break
    return dist1, index_i


def get_distributions(num_of_plans, num_of_actions, max_planning_time, mean, variance):
    padding = 10
    mpt = max(max_planning_time)
    dist = np.zeros((num_of_plans, num_of_actions, mpt + padding), dtype="float")
    planning_times = np.zeros((num_of_plans, num_of_actions), dtype="int")
    for i in range(0, num_of_plans):
        for j in range(0, num_of_actions):
            arr, index = get_single_distribution(max_planning_time[i], mean[i], variance[i],padding,mpt)
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
            e_dist[i][j] = s
            e_planning_times[i][j] = max_execution_time-1

    return e_dist,e_planning_times



if __name__ == "__main__":

    for t_ in range(4, 5):
        space_size = 0
        samples = 1
        max_iter = 1000
        cost_values = np.zeros((max_iter+2,samples+1), dtype="float")
        ctime = np.zeros((max_iter+2,samples+1), dtype="float")
        for sample_num in range(0, samples):
            print("SAMPLE : ",sample_num)
            m = [5, 10, 5]
            v = [2, 1, 3]
            num_of_plans = 2
            actions_per_plan = 2
            max_planning_time = np.array([3,3])
            deadline = 8
            actions = [1, 2]
            dist, planning_times = get_distributions(num_of_plans, actions_per_plan, max_planning_time, m, v)
            e_dist, e_times = get_execution_distributions(num_of_plans,actions_per_plan,max_execution_time=3)
            # print(dist)
            # print(e_dist)
            env = MetaWorldEnv(num_of_plans, actions_per_plan, deadline, actions, max_planning_time,
                               dist,planning_times,e_dist,e_times)
            mw = MetaReasoningWorld(env)
            print(env.successStates())

            # its, v, p, t = mw.do_value_iteration(100)
            env.print_for_me()
            print("DO Value Iteration")
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
            cost1 = 0.0
            seed_list = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45]
            runs = 100
            s = []
            for i in range(0, runs):
                pp, state_path, solution_cost = mw.get_policy_from_path(p)
                cost1 = cost1 + solution_cost
                s.append(solution_cost)
                # print(solution_cost)
                # print(pp)
            cost1 = cost1/runs
            print(np.mean(s))
            print(np.std(s))
            # print(st.t.interval(confidence=0.95, df=len(s) - 1, loc=np.mean(s), scale=st.sem(s)))
            ctime[0][sample_num] = t1
            cost_values[0][sample_num] = np.mean(s)
            print("DO MCTS")
            # Do MCTS
            # [2,5,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,1000]:
            list_k = []
            list_k_sd = []
            for k in [8000]:
                runs = 100
                cost_total = 0.0
                s = []
                for i in range(0,runs):
                    st_time = time.time()
                    curr_id = 0
                    pp = []
                    cost = 0.0
                    state_path = []
                    test = 0
                    while True :
                        cost = cost + env.reward_model[curr_id]
                        n = nd.Node(curr_id)
                        mcts = 0
                        mcts = MCTS(n, env, False)
                        best_action = mcts.Run(k)
                        pp.append(best_action)
                        state_path.append(env.get_state_from_id(curr_id))
                        res, _, _ = env.step(curr_id, best_action)
                        list1 = list(res.keys())
                        list2 = list(res.values())
                        curr_id = (random.choices(list1, weights=list2, k=1))[0]
                        if env.done(curr_id):
                            cost = cost + env.reward_model[curr_id]
                            break
                        test = test + 1
                    pp.pop()
                    cost_total = cost_total + cost
                    s.append(cost)
                list_k.append(np.mean(s))
                list_k_sd.append(np.std(s))
                t1 = time.time() - st_time
                # print(st.t.interval(confidence=0.95, df=len(s) - 1, loc=np.mean(s), scale=st.sem(s)))
                # ctime[k][sample_num] = t1
                # cost_values[k][sample_num] = np.mean(s)
            for i in range(0,len(list_k)):
                print(list_k[i])
            for i in range(0,len(list_k)):
                print(list_k_sd[i])
        avg = []
        moving_avg = []
        std = []

