from __future__ import division
from copy import deepcopy
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

# DEFAULT_DIST2 = [[[0.111, 0.823, 1.,    1.,    1.,    1.   ],
#                  [0.032, 0.452, 0.953, 1.,    1.,    1.   ]],
#                 [[0.039, 0.415, 0.923, 1.,    1.,    1.   ],
#                  [0.519, 1.,    1.,    1.,    1.,    1.   ]]]
#
# DEFAULT_TIMES2 = [[4, 3],
#                   [2, 4]]
# DEFAULT_EDIST = [[[0.24015373, 0.45464654, 0.30519973],
#                  [0.34614808, 0.3474152,  0.30643672]],
#
#                 [[0.31282752, 0.30373012, 0.38344236],
#                  [0.34851941, 0.30160948, 0.34987111]]]

DEFAULT_DIST2 = [[[0.072, 0.501, 0.904, 1., 1.],
                  [0.037, 0.333, 0.821, 0.986, 1.]],
                 [[0.014, 0.19, 0.717, 0.981, 1.],
                  [0.016, 0.702, 1., 1., 1.]]]

DEFAULT_TIMES2 = [[3, 3],
                  [4, 2]]
DEFAULT_EDIST = [[[0.24015373, 0.45464654, 0.30519973],
                  [0.34614808, 0.3474152, 0.30643672]],

                 [[0.31282752, 0.30373012, 0.38344236],
                  [0.34851941, 0.30160948, 0.34987111]]]

# DEFAULT_ETIMES = np.array([[2, 2], [2, 2]])
#
# DEFAULT_DIST2 = [[[0.0133, 0.44, 0.693, 0.88, 0.933, 0.9733, 0.9866, 1.0, 1.0, 1.0]],
#                  [[0.003, 0.004, 0.0153, 0.03076, 0.04615, 0.0615, 0.0923, 0.1384, 0.1692, 0.1896]]]
#
# DEFAULT_TIMES2 = [[8],
#                   [10]]
#
# DEFAULT_EDIST = [[[0.11595846, 0.15102481, 0.14918604, 0.18911111, 0.19425454, 0.20046504]],
#
#                  [[0.19869356, 0.22350258, 0.17317389, 0.14260042, 0.13689311,  0.12513644]]]
#
# DEFAULT_ETIMES = np.array([[5], [5]])


class MetaWorldEnv_v2:

    def __init__(self, num_of_plans_, actions_per_plan_, deadline_, actions_, max_planning_time_, verbose,
                 planning_dist_=DEFAULT_DIST2,
                 planning_times_=DEFAULT_TIMES2,
                 execution_dist_=DEFAULT_EDIST,
                 execution_times_=DEFAULT_ETIMES):

        # self.total_time = total_time_      #no need to have total_time; deadline is sufficient
        self.num_of_plans = num_of_plans_
        self.actions_per_plan = actions_per_plan_
        self.max_planning_time = max_planning_time_
        self.deadline = deadline_
        self.verbose = verbose
        self.planning_dist = planning_dist_
        self.planning_times = planning_times_
        self.execution_dist = execution_dist_
        self.execution_times = execution_times_
        self.num_of_actions = len(actions_)
        self.actions = actions_
        self.num_of_states = 0
        self.state_space = set()
        self.state_id_map = {}
        self.start_state = tuple([0] * (1 + 3 * self.num_of_plans))  # execution_time is stored

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

    def transition_function(self,state,action):
        act = action
        curr_st_id = self.add_state(state)
        if self.verbose:
            print("Generating transition states for state :", curr_st_id)
        state_id = []
        transition_prob = []
        curr_st_l = list(state)
        curr_time = curr_st_l[0]
        if self.verbose:
            print("Action ", act)
        next_st1_l = curr_st_l.copy()
        next_st2_l = curr_st_l.copy()
        next_st3_l = curr_st_l.copy()
        next_st1_l[0] = next_st1_l[0] + 1  # increase current time by 1
        next_st2_l[0] = next_st2_l[0] + 1  # increase current time by 1
        next_st3_l[0] = next_st3_l[0] + 1  # increase current time by 1

        index_last_action = 3 * act - 2
        index_pt = 3 * act - 1
        index_et = 3 * act

        if curr_time > self.deadline:
            state_id.append(curr_st_id)
            transition_prob.append(1.0)
            return dict(zip(state_id, transition_prob))

        last_refined_action = next_st1_l[index_last_action]  # next symbolic action
        pt_invested = next_st1_l[index_pt]
        exec_time = curr_st_l[index_et]
        t_prob = self.get_t_prob(act, last_refined_action, pt_invested)
        next_st1_l[index_pt] = next_st1_l[index_pt] + 1  # increase planning time for the first state
        next_st1 = tuple(next_st1_l)  # convert it into a tuple for the states list
        assert (next_st1_l[0] == curr_st_l[0] + 1)
        state_id = self.add_state(next_st1)  # append to the main set

        if self.verbose:
            print("Adding : ", curr_st_id, state_id, next_st1_l, 1.0 - t_prob)
        state_id.append(state_id)   # append the transition
        transition_prob.append(1.0 - t_prob)

        if next_st2_l[index_last_action] + 1 < self.actions_per_plan:  # add only valid state
            last_action = next_st2_l[index_last_action]
            next_st2_l[index_last_action] = next_st2_l[index_last_action] + 1  # next_action
            next_st2_l[index_pt] = 0  # make the planning time = 0

            # execution time invested so far
            execution_time_so_far = next_st2_l[index_et]

            next_st2i_l = next_st2_l.copy()
            met_last_action = self.execution_times[act - 1][last_action]

            for exec_time in range(1, met_last_action + 1):
                e_prob = self.execution_dist[act - 1][last_refined_action][exec_time]

                next_st2i_l = next_st2_l.copy()
                next_st2i_l[index_et] = execution_time_so_far + exec_time
                assert (next_st2_l[0] == curr_st_l[0] + 1)
                next_st2 = tuple(next_st2i_l)  # convert it into a tuple for the states list

                state_id = self.add_state(next_st2)
                if self.verbose:
                    print("Adding - - : ", curr_st_id, state_id, next_st2i_l, t_prob * e_prob)
                state_id.append(state_id)  # append the transition
                transition_prob.append(t_prob * e_prob)

        if next_st3_l[index_last_action] + 1 == self.actions_per_plan:  # add only valid state
            last_action = next_st3_l[index_last_action]
            execution_time_so_far = next_st3_l[index_et]

            next_st2i_l = next_st3_l.copy()
            met_last_action = self.execution_times[act - 1][last_action]

            for exec_time in range(1, met_last_action + 1):
                e_prob = self.execution_dist[act - 1][last_refined_action][exec_time]

                next_st2i_l = next_st3_l.copy()
                next_st2i_l[index_et] = execution_time_so_far + exec_time
                assert (next_st3_l[0] == curr_st_l[0] + 1)
                next_st2 = tuple(next_st2i_l)  # convert it into a tuple for the states list

                state_id = self.add_state(next_st2)
                if self.verbose:
                    print("Adding : ", curr_st_id, state_id, next_st2i_l, t_prob * e_prob)
                state_id.append(state_id)  # append the transition
                transition_prob.append(t_prob * e_prob)
        if abs(sum(transition_prob) - 0.0) < 0.001:
            state_id.append(curr_st_id)  # append the transition
            transition_prob.append(1.0)
        if self.is_terminal(curr_st_id) and abs(sum(transition_prob) - 0.0) < 0.001:
            state_id.append(curr_st_id)  # append the transition
            transition_prob.append(1.0)
        return dict(zip(state_id, transition_prob))

    # Calculates the Reward of being in a state
    # Argument : State Id
    # Returns : Reward
    def reward_function(self, state_id):
        state = self.get_state_from_id(state_id)
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

            if pt_invested > 0 and exec_time > 0 and curr_time + exec_time <= self.deadline and last_refined_action == last_action_id:
                r = r + 100
        return r

    # Take a step in the environment
    # Arguments : State Id, Action
    # Returns : Next State Ids, Transition Probs
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
        prob = list(self.transition_model[st_id][action - 1])
        ns = [i for i in range(0, self.num_of_states)]
        res = dict(zip(ns, prob))
        return res, ns, prob

    # Take a step in the environment
    # Arguments : State Id, Action
    # Returns : Sampled Next State Id, Reward
    def step_next_state(self, st_id, action):
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

    # Checks if the state is terminal
    # Arguments : State Id
    # Returns : True if it's a terminal state
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
            elif pt_invested > 0 and exec_time > 0 and curr_time + exec_time <= self.deadline and last_action_id == last_refined_action:
                return True
        return False

    def done(self, st_id):
        return self.is_terminal(st_id)

    def get_t_prob(self, act, last_refined_action, pt_invested):
        max_planning_time = len(self.planning_dist[act - 1][last_refined_action])
        if pt_invested >= max_planning_time:
            return 1.0
        else:
            return self.planning_dist[act - 1][last_refined_action][pt_invested]

    # To check if the transition matrix is stochastic (DEBUG CALL)
    def is_stochastic(self):
        n = self.num_of_states
        for i in range(0, n):
            for j in range(0, self.num_of_plans):
                if abs(sum(self.transition_model[i][j]) - 1.0) > 0.001:
                    return i, j, sum(self.transition_model[i][j])
        return 0, 0, 1.0

    def get_action_from_action_index(self, action_index):
        return self.actions[action_index]

    def print_for_me(self):
        print(self.planning_dist)

    def successStates(self):
        n = 0
        p = 0
        for i in range(0, self.num_of_states):
            if self.reward_model[i] > 0:
                n = n + 1
            if self.terminal_state[i]:
                p = p + 1

        return n, p, self.num_of_states

    def get_id_from_state(self, st):
        key = {i for i in self.state_id_map if self.state_id_map[i] == st}
        if len(key) > 0:
            assert (len(key) == 1)
            st_id = int(list(key)[0])
            return st_id

    def get_state_from_id(self, st_id):
        return self.state_id_map[st_id]
