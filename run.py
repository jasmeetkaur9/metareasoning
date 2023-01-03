from metaPlanning import MetaReasoningWorld
from metaWorldEnv import MetaWorldEnv, MetaWorldEnvM
import numpy as np


def get_single_distribution(max_planning_time, mean, variance):
    random_index = np.random.randint(1, max_planning_time)
    x = np.random.normal(loc=mean, scale=variance, size=1000)
    count, bins_count = np.histogram(x, bins=random_index + 1)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    dist1 = np.zeros(max_planning_time)
    index = 0
    for i in range(0, random_index):
        dist1[i] = cdf[i]
    for i in range(random_index, max_planning_time):
        dist1[i] = 1.0
    for i in range(0, max_planning_time):
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


if __name__ == "__main__":
    for t in range(6,7):
        ctime = 0.0
        space_size = 0
        samples = 1
        for i in range(0, samples):
            m = [5, 10]
            v = [2, 1]
            num_of_plans = 2
            actions_per_plan = 3
            max_planning_time = t
            total_time = (2 * 3 * max_planning_time) + 2
            deadline = (3 * max_planning_time)
            actions = [1, 2]
            dist, planning_times = get_distributions(num_of_plans, actions_per_plan, max_planning_time, m, v)
            print(dist)
            print(planning_times)
            # DEFAULT_DIST2 = [[[0.203, 0.878, 1.0, 1.0], [0.179, 0.847, 0.88, 0.9], [0.179, 0.847, 0.88, 0.9]],
            #                  [[0.174, 1.0, 1.0, 1.0], [0.126, 0.735, 1.0, 1.0], [0.179, 0.847, 1.0, 1.0]]]
            # DEFAULT_TIMES2 = np.array([[2, 5, 5], [1, 2, 2]])
            env = MetaWorldEnv(num_of_plans, actions_per_plan, deadline, actions, max_planning_time, dist,
                               planning_times)
            mw = MetaReasoningWorld(env)
            its, v, p, t = mw.do_value_iteration(100)
            print("Size of State Space ", env.num_of_states)
            print("Computation Time in secs ", t)
            print("Resultant policy", mw.get_policy_from_path(p))
            ctime = ctime + t
            space_size = space_size + env.num_of_states

        print("Avg State Size", space_size / samples)
        print("Avg Time", ctime / samples)
