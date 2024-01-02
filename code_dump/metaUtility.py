from __future__ import division
import numpy as np
import time
import random


def get_single_distribution(max_planning_time_, mean, variance, padding, mpt):
    x = np.random.normal(loc=mean, scale=variance, size=1000)
    index = random.randint(1, max_planning_time_)
    count, bins_count = np.histogram(x, index + 1)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    dist1 = np.zeros(mpt + padding)
    index_i = 0
    for i in range(0, index):
        dist1[i] = cdf[i]
    for i in range(index, mpt + padding):
        dist1[i] = 1.0
    for i in range(0, mpt + padding):
        if dist1[i] >= 1.0:
            index_i = i
            break
    return dist1, index_i


def get_distributions(num_of_plans, num_of_actions, max_planning_time, mean, variance):
    padding = 3
    mpt = max(max_planning_time)
    dist = np.zeros((num_of_plans, num_of_actions, mpt + padding), dtype="float")
    planning_times = np.zeros((num_of_plans, num_of_actions), dtype="int")
    for i in range(0, num_of_plans):
        for j in range(0, num_of_actions):
            arr, index = get_single_distribution(max_planning_time[i], mean[i], variance[i], padding, mpt)
            dist[i][j] = arr
            planning_times[i][j] = index

    return dist, planning_times


def get_execution_distributions(num_of_plans, num_of_actions, max_execution_time):
    mean = 20
    scale = 5
    e_dist = np.zeros((num_of_plans, num_of_actions, max_execution_time), dtype="float")
    e_planning_times = np.zeros((num_of_plans, num_of_actions), dtype="int")
    for i in range(0, num_of_plans):
        for j in range(0, num_of_actions):
            data = np.random.normal(loc=mean, scale=scale, size=max_execution_time)
            s = np.array(data)
            s = s / s.sum()
            e_dist[i][j] = s
            e_planning_times[i][j] = max_execution_time - 1

    return e_dist, e_planning_times


def round_robin_policy(time_steps,starting_plan,plans,max_time):
    p = []
    p.append(starting_plan)
    last_plan = starting_plan
    plan_index = 0
    total_plans = len(plans)
    for i in range(1,max_time):
        if i%time_steps == 0:
            plan_index = plan_index+1
            plan_index = plan_index % total_plans
            next_plan = plans[plan_index]
            p.append(next_plan)
            last_plan = next_plan
        else :
            p.append(last_plan)
    return p

def random_policy(plans,max_time):
    p = []
    for i in range(0,max_time):
        random_plan = np.random.choice(plans)
        p.append(random_plan)
    return p




