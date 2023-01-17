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

DEBUG = False
random.seed(40)


def log(s):
    if DEBUG:
        print(s)


MAX_STATES = 20000
num = 2
tm = np.zeros((MAX_STATES, num, MAX_STATES), dtype="float")

# DEFAULT_DIST2 = [[[0.05, 0.462, 0.943, 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                   [0.179, 0.863, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
#                   [0.044, 0.917, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
#
#                  [[0.018, 0.149, 0.526, 0.86, 0.992, 1., 1., 1., 1., 1., 1., 1.],
#                   [0.022, 0.235, 0.731, 0.968, 1., 1., 1., 1., 1., 1., 1., 1.],
#                   [0.012, 0.197, 0.683, 0.977, 1., 1., 1., 1., 1., 1., 1., 1.]]]
#
# DEFAULT_TIMES2 = np.array([[3, 2, 2], [5, 4, 4]])
# DEFAULT_EDIST = [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
#                  [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]
# DEFAULT_ETIMES = np.array([[1, 1, 1], [1, 1, 1]])


# DEFAULT_DIST2 = [[[0.041, 0.387, 0.891, 1., 1., 1., 1.],
#                               [0.52, 1., 1., 1., 1., 1., 1.]],
#
#                              [[0.078, 0.517, 0.94, 1., 1., 1., 1.],
#                               [0.065, 0.513, 0.953, 1., 1., 1., 1.]]]
# DEFAULT_TIMES2 = [[3, 1],
#                   [3, 3]]
#
# DEFAULT_EDIST = [[[0.5, 0.5], [0.5, 0.5]],
#                  [[0.5, 0.5], [0.5, 0.5]]]
# DEFAULT_ETIMES = np.array([[1, 1], [1, 1]])

DEFAULT_DIST2 = [[[0.036, 0.247, 0.648, 0.94, 0.997, 1., 1., 1., 1.],
                  [0.394, 1., 1., 1., 1., 1., 1., 1., 1.]],

                 [[0.015, 0.345, 0.909, 1., 1., 1., 1., 1., 1.],
                  [0.047, 0.719, 1., 1., 1., 1., 1., 1., 1.]]]
DEFAULT_TIMES2 = [[5, 1],
                  [3, 2]]

DEFAULT_EDIST = [[[0.5, 0.5], [0.5, 0.5]],
                 [[0.5, 0.5], [0.5, 0.5]]]
DEFAULT_ETIMES = np.array([[1, 1], [1, 1]])


class MetaWorldEnv:

    def __init__(self, num_of_plans_, actions_per_plan_, deadline_, actions_, max_planning_time_,
                 planning_dist_=DEFAULT_DIST2,
                 planning_times_=DEFAULT_TIMES2,
                 execution_dist_=DEFAULT_EDIST,
                 execution_times_=DEFAULT_ETIMES):

        # self.total_time = total_time_      #no need to have total_time; deadline is sufficient
        self.num_of_plans = num_of_plans_
        self.actions_per_plan = actions_per_plan_
        self.max_planning_time = max_planning_time_
        self.deadline = deadline_
        self.planning_dist = planning_dist_
        self.planning_times = planning_times_
        self.execution_dist = execution_dist_
        self.execution_times = execution_times_
        self.num_of_actions = len(actions_)
        self.actions = actions_
        self.num_of_states = 0
        self.state_space = set()
        self.state_id_map = {}
        self.get_state_space()
        self.transition_model = self.get_transition_model()
        self.start_state = tuple([0] * (1 + 3 * self.num_of_plans))  # execution_time is stored
        self.reward = self.get_reward_table()
        self.reward_model = self.get_reward_model()
        self.isStochasticR, self.isStochasticN, self.isStochasticS = self.is_stochastic()
        self.terminal_state = self.set_terminal()

    def add_state(self, s):
        if s not in self.state_space:
            self.state_space.add(s)
            self.state_id_map[self.num_of_states] = tuple(s)
            state_id = self.num_of_states
            self.num_of_states = self.num_of_states + 1
            return state_id
        else:
            state_id = {i for i in self.state_id_map if self.state_id_map[i] == s}
            assert (len(list(state_id)) == 1)
            st_id = int(list(state_id)[0])
            return st_id

    def get_state_id(self, s):
        state_id = {i for i in self.state_id_map if self.state_id_map[i] == s}
        st_id = int(list(state_id)[0])
        return st_id

    def get_state_space(self):
        shape_of_tuple = 1 + 3 * self.num_of_plans
        start = tuple([0] * shape_of_tuple)
        self.add_state(start)
        tmp = set()
        tmp.add(start)
        curr_time = 0
        while True:
            if len(tmp) <= 0:
                break
            curr_st = tmp.pop()
            curr_st_id = self.get_state_id(curr_st)
            curr_st_l = list(curr_st)
            curr_time = curr_st_l[0]
            # print("Curr state id : curr state :", curr_st_id,curr_st_l)
            for act in self.actions:
                tm[curr_st_id][act - 1] = 0.00
                # print("Action ", act)
                next_st1_l = curr_st_l.copy()
                next_st2_l = curr_st_l.copy()
                next_st3_l = curr_st_l.copy()
                next_st1_l[0] = next_st1_l[0] + 1  # increase current time by 1
                next_st2_l[0] = next_st2_l[0] + 1  # increase current time by 1
                next_st3_l[0] = next_st3_l[0] + 1  # increase current time by 1

                index_last_action = 3 * act - 2
                index_pt = 3 * act - 1
                index_et = 3 * act

                exec_time = curr_st_l[index_et]

                if curr_time > self.deadline:
                    tm[curr_st_id][act - 1][curr_st_id] = 1.0
                    continue

                last_refined_action = next_st1_l[
                    index_last_action]  # next symbolic action to transition depends on action
                pt_invested = next_st1_l[index_pt]
                t_prob = self.planning_dist[act - 1][last_refined_action][pt_invested]

                next_st1_l[index_pt] = next_st1_l[index_pt] + 1  # increase planning time for the first state
                next_st1 = tuple(next_st1_l)  # convert it into a tuple for the states list
                assert (next_st1_l[0] == curr_st_l[0] + 1)
                state_id = self.add_state(next_st1)  # append to the main set
                # print("Adding : ",curr_st_id,state_id,next_st1_l,1.0 - t_prob)
                tm[curr_st_id][act - 1][state_id] = (1.0 - t_prob)

                tmp.add(next_st1)  # append to the tmp set for loop
                # log(curr_st)
                log(next_st1)

                if next_st2_l[index_last_action] + 1 < self.actions_per_plan:  # add only valid state
                    log("Succ State-2 added")
                    last_action = next_st2_l[index_last_action]
                    next_st2_l[index_last_action] = next_st2_l[index_last_action] + 1  # next_action
                    next_st2_l[index_pt] = 0  # make the planning time = 0

                    #                     execution time invested so far
                    execution_time_so_far = next_st2_l[index_et]

                    next_st2i_l = next_st2_l.copy()
                    met_last_action = self.execution_times[act - 1][last_action]

                    for exec_time in range(0, met_last_action + 1):
                        e_prob = self.execution_dist[act - 1][last_refined_action][exec_time]

                        next_st2i_l = next_st2_l.copy()
                        next_st2i_l[index_et] = execution_time_so_far + exec_time
                        assert (next_st2_l[0] == curr_st_l[0] + 1)
                        next_st2 = tuple(next_st2i_l)  # convert it into a tuple for the states list

                        state_id = self.add_state(next_st2)
                        # print("Adding - - : ",curr_st_id,state_id,next_st2i_l,t_prob * e_prob )
                        tm[curr_st_id][act - 1][state_id] = t_prob * e_prob

                        tmp.add(next_st2)  # append to the tmp list for loop
                        log(next_st2)

                if next_st3_l[index_last_action] + 1 == self.actions_per_plan:  # add only valid state
                    log("Succ State-2 added")
                    last_action = next_st3_l[index_last_action]
                    execution_time_so_far = next_st3_l[index_et]

                    next_st2i_l = next_st3_l.copy()
                    met_last_action = self.execution_times[act - 1][last_action]

                    for exec_time in range(0, met_last_action + 1):
                        e_prob = self.execution_dist[act - 1][last_refined_action][exec_time]

                        next_st2i_l = next_st3_l.copy()
                        next_st2i_l[index_et] = execution_time_so_far + exec_time
                        assert (next_st3_l[0] == curr_st_l[0] + 1)
                        next_st2 = tuple(next_st2i_l)  # convert it into a tuple for the states list

                        state_id = self.add_state(next_st2)
                        # print("Adding -*- : ",curr_st_id,state_id,next_st2i_l,t_prob * e_prob )
                        tm[curr_st_id][act - 1][state_id] = t_prob * e_prob

                        tmp.add(next_st2)  # append to the tmp list for loop
                        log(next_st2)
                if abs(sum(tm[curr_st_id][act - 1]) - 0.0) < 0.001:
                    tm[curr_st_id][act - 1][curr_st_id] = 1.0
                if self.is_terminal(curr_st_id) and abs(sum(tm[curr_st_id][act - 1]) - 0.0) < 0.001:
                    tm[curr_st_id][act - 1][curr_st_id] = 1.0

    # this is very slow so need to come up with a better way
    def get_transition_model(self):
        n = self.num_of_states
        tm_tmp = np.zeros((n, self.num_of_plans, n), dtype="float")
        for i in range(0, n):
            for j in range(0, self.num_of_plans):
                for k in range(0, n):
                    tm_tmp[i][j][k] = float(f"{tm[i][j][k]}")
        return tm_tmp

    def step(self, st_id, action):
        assert (action in self.actions)
        assert (st_id in range(0, self.num_of_states))
        res = {}
        ns = []
        prob = []
        # for i in range(0, self.num_of_states):
        #     res[i] = self.transition_model[st_id][action - 1][i]
        #     ns.append(i)
        #     prob.append(res[i])
        prob = list(self.transition_model[st_id][action-1])
        ns = [i for i in range(0, self.num_of_states)]
        res = dict(zip(ns,prob))
        return res, ns, prob

    def step2(self, st_id, action):
        assert (action in self.actions)
        assert (st_id in range(0, self.num_of_states))
        res = {}
        ns = []
        prob = []
        reward = self.reward_model[st_id]
        # for i in range(0, self.num_of_states):
        #     res[i] = self.transition_model[st_id][action - 1][i]
        #     ns.append(i)
        #     prob.append(res[i])
        prob = list(self.transition_model[st_id][action - 1])
        ns = [i for i in range(0, self.num_of_states)]
        res = dict(zip(ns, prob))
        list1 = list(res.keys())
        list2 = list(res.values())
        curr_id = (random.choices(list1, weights=list2, k=1))[0]
        return curr_id, reward

    def get_id_from_state(self, st):
        key = {i for i in self.state_id_map if self.state_id_map[i] == st}
        if len(key) > 0:
            assert (len(key) == 1)
            st_id = int(list(key)[0])
            return st_id

    def get_state_from_id(self, st_id):
        return self.state_id_map[st_id]

    def is_terminal(self, st_id):
        state = self.get_state_from_id(st_id)
        state_l = list(state)
        for j in range(1, self.num_of_plans + 1):
            plan_id = j - 1
            last_refined_action = state_l[3 * j - 2]
            pt_invested = state_l[3 * j - 1]
            exec_time = state_l[3 * j]
            curr_time = state_l[0]
            last_action_id = self.actions_per_plan - 1
            if curr_time > self.deadline:
                return True
            elif exec_time > 0 and curr_time + exec_time <= self.deadline and last_action_id == last_refined_action:
                return True
        return False

    def set_terminal(self):
        arr = [False for i in range(0, self.num_of_states)]
        for i in range(0, self.num_of_states):
            state = self.get_state_from_id(i)
            state_l = list(state)
            for j in range(1, self.num_of_plans + 1):
                plan_id = j - 1
                last_refined_action = state_l[3 * j - 2]
                pt_invested = state_l[3 * j - 1]
                exec_time = state_l[3 * j]
                curr_time = state_l[0]
                last_action_id = self.actions_per_plan - 1
                if curr_time > self.deadline:
                    arr[i] = True
                elif exec_time > 0 and curr_time + exec_time <= self.deadline and last_action_id == last_refined_action:
                    arr[i] = True

        return arr

    def done(self, st_id):
        return self.terminal_state[st_id]

    def get_reward_table(self):
        reward = {}
        for i in range(0, self.num_of_states):
            state = self.get_state_from_id(i)
            state_l = list(state)
            r = 0.0
            for j in range(1, self.num_of_plans + 1):
                plan_id = j - 1
                last_refined_action = state_l[3 * j - 2]
                pt_invested = state_l[3 * j - 1]
                exec_time = state_l[3 * j]
                last_action_id = self.actions_per_plan - 1

                t_prob = self.planning_dist[plan_id][last_action_id][pt_invested]
                curr_time = state_l[0]

                if exec_time > 0 and curr_time + exec_time <= self.deadline and last_refined_action == last_action_id:
                    r = 100.0
            reward[i] = r
        return reward

    def get_reward_model(self):
        result = self.reward.values()
        data = list(result)
        model = np.array(data)
        return model

    def is_stochastic(self):
        n = self.num_of_states
        for i in range(0, n):
            for j in range(0, self.num_of_plans):
                if abs(sum(self.transition_model[i][j]) - 1.0) > 0.001:
                    return i, j, sum(self.transition_model[i][j])
        return 0, 0, 1.0

    def get_action_from_action_index(self, action_index):
        return self.actions[action_index]