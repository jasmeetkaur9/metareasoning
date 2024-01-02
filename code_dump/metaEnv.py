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

# DEFAULT_DIST2 = [[[0.072, 0.501, 0.904, 1., 1.],
#                   [0.037, 0.333, 0.821, 0.986, 1.]],
#                  [[0.014, 0.19, 0.717, 0.981, 1.],
#                   [0.016, 0.702, 1., 1., 1.]]]
#
# DEFAULT_TIMES2 = [[3, 3],
#                   [4, 2]]
# DEFAULT_EDIST = [[[0.24015373, 0.45464654, 0.30519973],
#                   [0.34614808, 0.3474152, 0.30643672]],
#
#                  [[0.31282752, 0.30373012, 0.38344236],
#                   [0.34851941, 0.30160948, 0.34987111]]]
#
# DEFAULT_ETIMES = np.array([[2, 2], [2, 2]])

# DEFAULT_DIST2 = [[[0.0133, 0.44, 0.693, 0.88, 0.933, 0.9733, 0.9866, 1.0, 1.0, 1.0]],
#                  [[0.003, 0.004, 0.0153, 0.03076, 0.04615, 0.0615, 0.0923, 0.1384, 0.1692, 0.1896]]]

# DEFAULT_TIMES2 = [[8],
#                   [10]]

# DEFAULT_EDIST = [[[0.11595846, 0.15102481, 0.14918604, 0.18911111, 0.19425454, 0.20046504]],

#                  [[0.19869356, 0.22350258, 0.17317389, 0.14260042, 0.13689311,  0.12513644]]]

# DEFAULT_ETIMES = np.array([[5], [5]])

# DEFAULT_DIST2 = [[[0.02, 0.02369, 0.02369, 0.34308, 0.617268, 0.78782,0.9112, 0.95420, 0.977659, 0.98666, 1.0]],
#                  [[0.106972,0.106972,0.278794,0.450616,0.5332514,0.5997631,0.6650907,0.7149744,0.764858,0.8081900,0.835903]]]


# DEFAULT_TIMES2 = [[11],[12]]

# DEFAULT_EDIST = [[[0.001,0.001, 0.001,0.001, 0.056869, 0.032497, 0.001,0.001]],
#                  [[0.001,0.001, 0.001,0.0330971, 0.001, 0.001, 0.001,0.001]]]

# DEFAULT_ETIMES = np.array([[7], [7]])

DEFAULT_DIST2 = [[[0.02, 0.02369, 0.02369, 0.34308, 0.617268, 0.78782,0.9112, 0.95420, 0.977659, 0.98666, 1.0],
                  [0.02, 0.02369, 0.02369, 0.34308, 0.617268, 0.78782,0.9112, 0.95420, 0.977659, 0.98666, 1.0]],
                 [[0.106972,0.106972,0.278794,0.450616,0.5332514,0.5997631,0.6650907,0.7149744,0.764858,0.8081900,0.835903],
                  [0.106972,0.106972,0.278794,0.450616,0.5332514,0.5997631,0.6650907,0.7149744,0.764858,0.8081900,0.835903]]]


DEFAULT_TIMES2 =  [[11, 11],
                   [12, 12]]

DEFAULT_EDIST = [[[0.001,0.001, 0.001,0.001, 0.056869, 0.032497, 0.001,0.001],
                  [0.001,0.001, 0.001,0.001, 0.056869, 0.032497, 0.001,0.001]],
                 [[0.001,0.001, 0.001,0.0330971, 0.001, 0.001, 0.001,0.001],
                  [0.001,0.001, 0.001,0.0330971, 0.001, 0.001, 0.001,0.001]]]

DEFAULT_ETIMES = np.array([[7, 7], [7, 7]])

# DEFAULT_DIST2 = [[[0.0133, 0.44, 0.693, 0.88, 0.933, 0.9733, 0.9866, 1.0, 1.0, 1.0],
#                   [0.0133, 0.44, 0.693, 0.88, 0.933, 0.9733, 0.9866, 1.0, 1.0, 1.0]],
#                  [[0.003, 0.004, 0.0153, 0.03076, 0.04615, 0.0615, 0.0923, 0.1384, 0.1692, 0.1896],
#                   [0.003, 0.004, 0.0153, 0.03076, 0.04615, 0.0615, 0.0923, 0.1384, 0.1692, 0.1896]]]

# DEFAULT_TIMES2 = [[8, 8],
#                   [10, 10]]

# DEFAULT_EDIST = [[[0.11595846, 0.15102481, 0.14918604, 0.18911111, 0.19425454, 0.20046504],
#                   [0.11595846, 0.15102481, 0.14918604, 0.18911111, 0.19425454, 0.20046504]],

#                  [[0.19869356, 0.22350258, 0.17317389, 0.14260042, 0.13689311,  0.12513644],
#                   [0.19869356, 0.22350258, 0.17317389, 0.14260042, 0.13689311,  0.12513644]]]

# DEFAULT_ETIMES = np.array([[5, 5], [5, 5]])



class MetaWorldEnv:

    def __init__(self, num_of_plans_, actions_per_plan_, deadline_, actions_, max_planning_time_, verbose, version1, 
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
        self.version1 = version1

    def reset(self):
        self.num_of_states = 0
        self.state_space = set()
        self.state_id_map = {}
        self.start_state = tuple([0] * (1 + 3 * self.num_of_plans))  # execution_time is stored
        if self.version1 :
            self.get_state_space()
            self.transition_model = self.get_transition_model()
            self.reward = self.get_reward_table()
            self.reward_model = self.get_reward_model()
            self.isStochasticR, self.isStochasticN, self.isStochasticS = self.is_stochastic()
            self.terminal_state = self.set_terminal()
        else :
            shape_of_tuple = 1 + 3 * self.num_of_plans
            start = tuple([0] * shape_of_tuple)
            self.add_state(start)

    def add_state(self, s):
        if s not in self.state_space:
            self.state_space.add(s)
            self.state_id_map[self.num_of_states] = tuple(s)
            state_id = self.num_of_states
            self.num_of_states = self.num_of_states + 1
            return state_id
        else:
            # Slow : Access in O(n)
            # state_id = {i for i in self.state_id_map if self.state_id_map[i] == s}
            # assert (len(list(state_id)) == 1)
            # st_id_old = int(list(state_id)[0])

            # Use the following : Access in O(1)
            key_list = list(self.state_id_map.keys())
            val_list = list(self.state_id_map.values())
            position = val_list.index(s)  
            state_id = key_list[position]
            st_id = int(state_id) 
            return st_id

    def get_state_id(self, s):
        # Slow : Access in O(n)
        # state_id = {i for i in self.state_id_map if self.state_id_map[i] == s}
        # st_id = int(list(state_id)[0])

        # Use the following : Access in O(1)
        key_list = list(self.state_id_map.keys())
        val_list = list(self.state_id_map.values())
        position = val_list.index(s)  
        state_id = key_list[position]
        st_id = int(state_id) 
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
            if self.verbose:
                print("Curr state id : curr state :", curr_st_id, curr_st_l)
            for act in self.actions:
                tm[curr_st_id][act - 1] = 0.00
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
                    tm[curr_st_id][act - 1][curr_st_id] = 1.0
                    continue

                last_refined_action = next_st1_l[index_last_action]  # next symbolic action
                pt_invested = next_st1_l[index_pt]
                exec_time = curr_st_l[index_et]
                t_prob = self.get_t_prob(act,last_refined_action,pt_invested)

                next_st1_l[index_pt] = next_st1_l[index_pt] + 1  # increase planning time for the first state
                next_st1 = tuple(next_st1_l)  # convert it into a tuple for the states list
                assert (next_st1_l[0] == curr_st_l[0] + 1)
                state_id = self.add_state(next_st1)  # append to the main set

                if self.verbose:
                    print("Adding : ", curr_st_id, state_id, next_st1_l, 1.0 - t_prob)

                tm[curr_st_id][act - 1][state_id] = (1.0 - t_prob)
                tmp.add(next_st1)  # append to the tmp set for loop

                if next_st2_l[index_last_action] + 1 < self.actions_per_plan:  # add only valid state
                    last_action = next_st2_l[index_last_action]
                    next_st2_l[index_last_action] = next_st2_l[index_last_action] + 1  # next_action
                    next_st2_l[index_pt] = 0  # make the planning time = 0

                    # execution time invested so far
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
                        if self.verbose:
                            print("Adding - - : ", curr_st_id, state_id, next_st2i_l, t_prob * e_prob)

                        tm[curr_st_id][act - 1][state_id] = t_prob * e_prob
                        tmp.add(next_st2)  # append to the tmp list for loop

                if next_st3_l[index_last_action] + 1 == self.actions_per_plan:  # add only valid state
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
                        if self.verbose:
                            print("Adding : ", curr_st_id, state_id, next_st2i_l, t_prob * e_prob)
                        tm[curr_st_id][act - 1][state_id] = t_prob * e_prob

                        tmp.add(next_st2)  # append to the tmp list for loop
                        log(next_st2)

                # sanity checks
                if abs(sum(tm[curr_st_id][act - 1]) - 0.0) < 0.001:
                    tm[curr_st_id][act - 1][curr_st_id] = 1.0
                if self.is_terminal(curr_st_id) and abs(sum(tm[curr_st_id][act - 1]) - 0.0) < 0.001:
                    tm[curr_st_id][act - 1][curr_st_id] = 1.0

    # TODO: this is very slow so need to come up with a better way
    # returns copy of transition model (DEBUG CALL)
    def get_transition_model(self):
        n = self.num_of_states
        tm_tmp = np.zeros((n, self.num_of_plans, n), dtype="float")
        for i in range(0, n):
            for j in range(0, self.num_of_plans):
                for k in range(0, n):
                    tm_tmp[i][j][k] = float(f"{tm[i][j][k]}")
        return tm_tmp

    # This call the states that have already been generated
    def transition_function(self,st_id,action):
        act = action
        state = self.get_state_from_id(st_id)
        curr_st_id = self.add_state(state)
        if self.verbose:
            print("Generating transition states for state :", curr_st_id)
        state_id_list = []
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
            state_id_list.append(curr_st_id)
            transition_prob.append(1.0)
            return dict(zip(state_id_list, transition_prob))

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

        state_id_list.append(state_id)   # append the transition
        transition_prob.append(1.0 - t_prob)

        if next_st2_l[index_last_action] + 1 < self.actions_per_plan:  # add only valid state
            last_action = next_st2_l[index_last_action]
            next_st2_l[index_last_action] = next_st2_l[index_last_action] + 1  # next_action
            next_st2_l[index_pt] = 0  # make the planning time = 0

            # execution time invested so far
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
                if self.verbose:
                    print("Adding - - : ", curr_st_id, state_id, next_st2i_l, t_prob * e_prob)
                
                state_id_list.append(state_id)  # append the transition
                transition_prob.append(t_prob * e_prob)

        if next_st3_l[index_last_action] + 1 == self.actions_per_plan:  # add only valid state
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
                if self.verbose:
                    print("Adding : ", curr_st_id, state_id, next_st2i_l, t_prob * e_prob)
                state_id_list.append(state_id)  # append the transition
                transition_prob.append(t_prob * e_prob)
        if abs(sum(transition_prob) - 0.0) < 0.001:
            state_id_list.append(curr_st_id)  # append the transition
            transition_prob.append(1.0)
        if self.is_terminal(curr_st_id) and abs(sum(transition_prob) - 0.0) < 0.001:
            state_id_list.append(curr_st_id)  # append the transition
            transition_prob.append(1.0)
        return dict(zip(state_id_list, transition_prob))

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
        if self.version1 :
            prob = list(self.transition_model[st_id][action - 1])
            ns = [i for i in range(0, self.num_of_states)]
            res = dict(zip(ns, prob))
        else :
            res = self.transition_function(st_id,action)
            ns = list(res.keys())
            prob = list(res.values())
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
        curr_id = st_id
        reward = 0.0
        if self.version1:
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
        else :
            reward = self.reward_function(st_id)
            res = self.transition_function(st_id,action)
            ns = list(res.keys())
            prob = list(res.values())
            curr_id = (random.choices(ns, weights=prob, k=1))[0]
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

    # Creates an array of states and their termination status
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
                elif pt_invested > 0 and exec_time > 0 and curr_time + exec_time <= self.deadline and last_action_id == last_refined_action:
                    arr[i] = True
        return arr

    def done(self, st_id):
        if self.version1 :
            return self.terminal_state[st_id]
        else :
            return self.is_terminal(st_id)

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
                curr_time = state_l[0]
                if pt_invested > 0 and exec_time > 0 and curr_time + exec_time <= self.deadline and last_refined_action == last_action_id:
                    r = 100
            reward[i] = r
        return reward

    def get_reward_model(self):
        result = self.reward.values()
        data = list(result)
        model = np.array(data)
        return model

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
            curr_time = state_l[0]

            if pt_invested > 0 and exec_time > 0 and curr_time + exec_time <= self.deadline and last_refined_action == last_action_id:
                r = r + 100
        return r

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
                if self.version1 :
                    if abs(sum(self.transition_model[i][j]) - 1.0) > 0.001:
                        return i, j, sum(self.transition_model[i][j])
                else :
                    res = self.transition_function(i,j)
                    if abs(sum(res) - 1.0) > 0.001:
                        return i, j, sum(res)
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