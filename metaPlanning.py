from metaWorldEnv import MetaWorldEnv
from metaWorldEnv import MetaWorldEnvM
import numpy as np
import time


class MetaReasoningWorld:
    def __init__(self, env):
        self.env = env

    def do_value_iteration(self, max_iterations):
        start_time = time.time()
        env = self.env
        max_iter = max_iterations  # Maximum number of iterations
        theta = 0.0001
        n = env.num_of_states
        Value_Table = np.random.rand(n, 1)
        # Value_Table = np.zeros((n,1),dtype="float")
        reward_table = env.reward
        # print(reward_table)
        gamma = 0.9
        pi = {}
        i = 0

        for i in range(max_iter):
            # print(i,"***************************************************")
            max_diff = 0  # Initialize max difference
            Value_Table_New = np.zeros((n, 1), dtype="float")
            delta = 0.0
            for st_id in range(0, env.num_of_states):
                max_val = 0
                r_value = reward_table[st_id]
                best_action = 0  # include do nothing

                for action in env.actions:

                    res = env.step(st_id, action)
                    val = r_value

                    for next_state in res:
                        n_st, prob = next_state
                        if n_st != st_id:
                            val = val + (prob * (gamma * Value_Table[n_st]))

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
        shape_of_tuple = 1 + 2 * self.env.num_of_plans
        start = tuple([0] * shape_of_tuple)
        st_id = self.env.get_id_from_state(start)
        curr_time = 0
        curr_id = st_id
        policy = []
        state_path = []

        while curr_time <= self.env.deadline:
            state_path.append(self.env.get_state_from_id(curr_id))
            curr_state_ls = list(self.env.get_state_from_id(curr_id))
            next_action = pi[curr_id]
            policy.append(next_action)
            res = self.env.step(curr_id, next_action)

            if len(res) == 0 or (len(res) == 1 and abs(list(res[0])[1] - 0.0) < 0.001):  # ADDED TERMINAL
                break
            next_states = []
            probs = []
            for i in range(0, len(res)):
                next_states.append(res[i][0])
                probs.append(res[i][1])

            # curr_id = list(res[0])[0]
            curr_id = (np.random.choice(next_states, 1, p=probs))[0]
            curr_time = (list(self.env.get_state_from_id(curr_id)))[0]
        return policy[:-1], state_path



