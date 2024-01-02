from __future__ import division
import numpy as np
import time
import random


class MetaReasoningWorld:
    def __init__(self, env, verbose):
        self.env = env
        self.verbose = verbose

    def do_value_iteration(self, max_iterations):
        start_time = time.time()
        env = self.env
        max_iter = max_iterations  # Maximum number of iterations
        theta = 0.0001
        n = env.num_of_states
        # Value_Table = np.random.rand(n, 1)
        Value_Table = np.zeros((n), dtype="float")
        reward_table = env.reward
        # print(reward_table)
        gamma = 0.9
        pi = {}
        i = 0

        for i in range(max_iter):
            if self.verbose:
                print("Iteration : ",i)
            max_diff = 0  # Initialize max difference
            Value_Table_New = np.zeros((n), dtype="float")

            delta = 0.0

            for st_id in range(0, env.num_of_states):
                max_val = 0
                r_value = reward_table[st_id]
                best_action = 0  # include do nothing

                for action in env.actions:

                    res, ns, prob = env.step(st_id, action)
                    val = 0.0
                    pt = 0.0
                    reward_array = []
                    reward_array = np.full(self.env.num_of_states, r_value)
                    val = sum(np.multiply(np.array(prob), reward_array + gamma * Value_Table))

                    #                     for ns,prob in res.items():
                    #                         val = val + (prob * (r_value + gamma * Value_Table[int(ns)]))
                    #                         pt = pt + prob

                    if max_val <= val:
                        max_val = val
                        best_action = action

                if Value_Table_New[st_id] <= max_val:
                    pi[st_id] = best_action
                    Value_Table_New[st_id] = max_val
                delta = max(delta, abs(Value_Table[st_id] - Value_Table_New[st_id]))

            Value_Table = Value_Table_New
            if delta < theta:
                time_invested = (time.time() - start_time)
                return i, Value_Table, pi, time_invested  # ADDED DELTA
        time_invested = (time.time() - start_time)
        return (max_iterations - 1), Value_Table, pi, time_invested

    def get_policy_from_path(self, pi):
        shape_of_tuple = 1 + 3 * self.env.num_of_plans
        start = tuple([0] * shape_of_tuple)
        st_id = self.env.get_id_from_state(start)
        t = 0
        curr_id = st_id
        policy = []
        state_path = []
        cost = 0.0
        while True:
            state_path.append(self.env.get_state_from_id(curr_id))
            curr_state_ls = list(self.env.get_state_from_id(curr_id))
            cost = cost + self.env.reward_model[curr_id]
            next_action = pi[curr_id]
            policy.append(next_action)
            res, _, _ = self.env.step(curr_id, next_action)
            if self.env.done(curr_id):
                break
            list1 = list(res.keys())
            list2 = list(res.values())
            curr_id = (random.choices(list1, weights=list2, k=1))[0]
        policy.pop()
        return policy, state_path, cost

    def get_solution_using_policy(self, p):
        shape_of_tuple = 1 + 3 * self.env.num_of_plans
        start = tuple([0] * shape_of_tuple)
        st_id = self.env.get_id_from_state(start)
        t = 0
        curr_id = st_id
        total_reward = 0.0
        index = -1
        state_path = []
        while index < len(p)-1:
            index = index + 1
            state_path.append(self.env.get_state_from_id(curr_id))
            # total_reward = total_reward + self.env.reward_model[curr_id]
            total_reward = total_reward + self.env.reward_function(curr_id)
            next_action = p[index]
            res, _, _ = self.env.step(curr_id, next_action)
            if self.env.done(curr_id):
                break
            list1 = list(res.keys())
            list2 = list(res.values())
            curr_id = (random.choices(list1, weights=list2, k=1))[0]

        return state_path, total_reward

