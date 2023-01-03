import numpy as np
from itertools import count
from collections import defaultdict
import collections

DEBUG = False


def log(s):
    if DEBUG:
        print(s)


# This is a MetaWorldEnv class where the inputs are defined in the functions manually.
class MetaWorldEnvM:
    def __init__(self):
        self.total_time = 22
        self.num_of_plans = 3
        self.actions_per_plan = 1
        #self.max_planning_time = 7     #no need to have total_planning time; deadline is sufficient
        self.deadline = 10
        self.planning_dist = [[[0.001, 0.8, 0.95, 0.99, 0.99, 0.99, 0.99]],
                              [[0.001, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
                              [[0.01, 0.2, 0.4, 0.5, 0.7, 0.8, 0.86]]]
        self.planning_times = np.array([[3], [1], [8]])
        self.actions = [1, 2, 3]
        self.state_space = self.get_state_space()
        self.state_id_map = self.get_state_id_map()
        self.state_ids = self.get_state_ids()
        self.num_of_states = len(self.state_ids)
        self.start_state = tuple([0] * (1 + 2 * self.num_of_plans))
        self.reward = self.get_reward_table()

    def get_state_space(self):
        shape_of_tuple = 1 + 2 * self.num_of_plans
        start = tuple([0] * shape_of_tuple)
        # state_space_list = []                  #Use set instead of list to get rid of storing duplicates
        state_space_set = set()
        # state_space_list.append(start)
        state_space_set.add(start)
        tmp = set()
        # state_space_list.append(start)
        tmp.add(start)
        curr_time = 0
        while True:
            if len(tmp) <= 0:
                break
            curr_st = tmp.pop()
            curr_st_l = list(curr_st)
            curr_time = curr_st_l[0]
            if curr_time > self.deadline:
                break

            log("Current State is")
            log(curr_st)
            log("Current time :---------------")
            log(curr_time)
            log(self.deadline)

            for act in self.actions:
                log("Current Action is")
                log(act)

                next_st1_l = curr_st_l.copy()
                next_st2_l = curr_st_l.copy()
                next_st1_l[0] = next_st1_l[0] + 1  # increase current time by 1
                next_st2_l[0] = next_st2_l[0] + 1  # increase current time by 1

                index_last_action = 2 * act - 1
                index_pt = 2 * act

                if next_st1_l[index_pt] + 1 < self.max_planning_time:  # add only valid state
                    log("Succ State-1 added")
                    next_st1_l[index_pt] = next_st1_l[index_pt] + 1  # increase planning time for the first state
                    next_st1 = tuple(next_st1_l)  # convert it into a tuple for the states list

                    state_space_set.add(next_st1)  # append to the main set
                    tmp.add(next_st1)  # append to the tmp set for loop
                    # log(curr_st)
                    log(next_st1)

                if next_st2_l[index_last_action] + 1 < self.actions_per_plan:  # add only valid state
                    log("Succ State-2 added")
                    next_st2_l[index_last_action] = next_st2_l[index_last_action] + 1  # next_action
                    next_st2_l[index_pt] = 0  # make the planning time = 0
                    next_st2 = tuple(next_st2_l)  # convert it into a tuple for the states list

                    state_space_set.add(next_st2)  # append to the main list
                    tmp.add(next_st2)  # append to the tmp list for loop
                    # log(curr_st)
                    log(next_st2)

            log("Succ State-3 added")
            next_st3_l = curr_st_l.copy()
            next_st3_l[0] = next_st3_l[0] + 1
            next_st3 = tuple(next_st3_l)
            state_space_set.add(next_st3)
            tmp.add(next_st3)

            log(curr_st)
            log(next_st3)

        log("The States List with Duplicates has")
        # state_space_list.reverse()
        log(len(state_space_set))
        return state_space_set

    def get_state_id_map(self):
        mapping = defaultdict(count().__next__)
        result = []
        for element in self.state_space:
            result.append(mapping[tuple(element)])

        res = dict(zip(result, self.state_space))
        res = collections.OrderedDict(sorted(res.items()))
        return res

    def get_state_ids(self):
        mapping = defaultdict(count().__next__)
        result = []
        for element in self.state_space:
            result.append(mapping[tuple(element)])

        return result

    def isStatePresent(self, st):
        key = {i for i in self.state_id_map if self.state_id_map[i] == st}
        if len(key) > 0:
            assert (len(key) == 1)
            return True
        return False

    # This function takes the state id and action and returns a list of size 0-2
    # of tuples where every tuple is (next_state_id, prob)
    def step(self, st_id, action):
        assert (action in self.actions)
        assert (st_id in self.state_ids)
        curr_st = self.state_id_map[st_id]
        curr_st_l = list(curr_st)
        res = []
        next_st1_l = curr_st_l.copy()
        next_st2_l = curr_st_l.copy()
        next_st3_l = curr_st_l.copy()
        next_st1_l[0] = next_st1_l[0] + 1  # increase current time by 1
        next_st2_l[0] = next_st2_l[0] + 1  # increase current time by 1
        next_st3_l[0] = next_st3_l[0] + 1  # increase current time by 1

        next_st3 = tuple(next_st3_l)
        key3 = {i for i in self.state_id_map if self.state_id_map[i] == next_st3}

        index_last_action = 2 * action - 1
        index_pt = 2 * action

        last_refined_action = next_st1_l[index_last_action]  # next symbolic action to transition depends on action
        pt_invested = next_st1_l[index_pt]
        t_prob = self.planning_dist[action - 1][last_refined_action][pt_invested]

        next_st1_l[index_pt] = next_st1_l[index_pt] + 1  # increase planning time for the first state
        next_st1 = tuple(next_st1_l)  # convert it into a tuple for the states list
        key1 = {i for i in self.state_id_map if self.state_id_map[i] == next_st1}

        #         if len(key1)>0 :               #the transition prob must some to 1.0 otherwise incorrect results
        #             assert(len(key1)==1)
        #             next_st1_id = int(list(key1)[0])
        #             res.append((next_st1_id,float(1-t_prob)))

        next_st2_l[index_last_action] = next_st2_l[index_last_action] + 1  # next_action
        next_st2_l[index_pt] = 0  # make the planning time = 0
        next_st2 = tuple(next_st2_l)  # convert it into a tuple for the states list
        key2 = {i for i in self.state_id_map if self.state_id_map[i] == next_st2}  # since you are looking only in
        # set of valid states, you don't need to check it again here

        #         if len(key2)>0 :                #the transition prob must some to 1.0 otherwise incorrect results
        #             assert(len(key2)==1)
        #             next_st2_id = int(list(key2)[0])
        #             res.append((next_st2_id,float(t_prob)))

        #         if len(res) == 0:
        #             res.append((st_id,0.00))

        if len(key1) > 0 and len(key2) > 0:
            assert (len(key1) == 1)
            assert (len(key2) == 1)
            next_st1_id = int(list(key1)[0])
            res.append((next_st1_id, float(1 - t_prob)))
            next_st2_id = int(list(key2)[0])
            res.append((next_st2_id, float(t_prob)))
        elif len(key1) > 0:
            next_st1_id = int(list(key1)[0])
            res.append((next_st1_id, float(0.99)))
        elif len(key2) > 0:
            next_st2_id = int(list(key2)[0])
            res.append((next_st2_id, float(0.99)))
        elif len(key3) > 0:
            assert (len(key3) == 1)
            next_st3_id = int(list(key3)[0])
            res.append((next_st3_id, 0.00))

        return res

    def get_id_from_state(self, st):
        key = {i for i in self.state_id_map if self.state_id_map[i] == st}
        if len(key) > 0:
            assert (len(key) == 1)
            st_id = int(list(key)[0])
            return st_id

    def get_state_from_id(self, st_id):
        return self.state_id_map[st_id]

    def get_reward_table(self):
        reward = {}
        for i in range(0, self.num_of_states):
            state = self.get_state_from_id(i)
            state_l = list(state)
            r = 0.0
            for j in range(1, self.num_of_plans + 1):
                plan_id = j - 1
                last_refined_action = state_l[2 * j - 1]
                pt_invested = state_l[2 * j]
                # t_prob =  self.planning_dist[plan_id][last_refined_action][pt_invested]
                # it should not be the last action refined because that does not mean the plan is over
                last_action_id = self.actions_per_plan - 1
                t_prob = self.planning_dist[plan_id][last_action_id][pt_invested]
                curr_time = state_l[0]
                if t_prob >= 0.9 and curr_time <= self.deadline and pt_invested == self.planning_times[plan_id][last_action_id]:
                    r = r + 1.0
            reward[i] = r
        return reward


# Assumptions on the input :
# Deadline is global and is over all the plans.
# Number of actions per plan is fixed. TO DO: change it to variable.
# Max_planning_time is also fixed for all the actions in very plan. TO DO : change this to variable as well
# Actions are the plans. They do not represent symbolic actions.

# This class defines MetaWorldEnv that takes the input from the user. Some defaults are given.
DEFAULT_DIST1 = [[[0.1, 0.5, 0.6, 0.6]], [[0.1, 0.8, 0.99, 0.99]], [[0.1, 0.99, 0.99, 0.99]]]
DEFAULT_TIMES1 = np.array([[[4]], [[2]], [[1]]])

DEFAULT_DIST2 = [[[0.001, 0.8, 0.95, 0.99, 0.99, 0.99, 0.99]],
                 [[0.001, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
                 [[0.01, 0.2, 0.4, 0.5, 0.7, 0.8, 0.86]]]
DEFAULT_TIMES2 = np.array([[3], [1], [8]])


class MetaWorldEnv:
    def __init__(self, num_of_plans_, actions_per_plan_, deadline_, actions_, max_planning_time_,
                 planning_dist_=DEFAULT_DIST2,
                 planning_times_=DEFAULT_TIMES2):

        #self.total_time = total_time_      #no need to have total_time; deadline is sufficient
        self.num_of_plans = num_of_plans_
        self.actions_per_plan = actions_per_plan_
        self.max_planning_time = max_planning_time_
        self.deadline = deadline_
        self.planning_dist = planning_dist_
        self.planning_times = planning_times_
        self.num_of_actions = len(actions_)
        self.actions = actions_
        self.state_space = self.get_state_space()
        self.state_id_map = self.get_state_id_map()
        self.state_ids = self.get_state_ids()
        self.num_of_states = len(self.state_ids)
        self.start_state = tuple([0] * (1 + 2 * self.num_of_plans))
        self.reward = self.get_reward_table()

    def get_state_space(self):
        shape_of_tuple = 1 + 2 * self.num_of_plans
        start = tuple([0] * shape_of_tuple)
        # state_space_list = []
        state_space_set = set()
        # state_space_list.append(start)
        state_space_set.add(start)
        tmp = set()
        # state_space_list.append(start)
        tmp.add(start)
        curr_time = 0
        while (True):
            if len(tmp) <= 0:
                break
            curr_st = tmp.pop()
            curr_st_l = list(curr_st)
            curr_time = curr_st_l[0]
            if (curr_time > self.deadline):
                break

            log("Current State is")
            log(curr_st)
            log("Current time :---------------")
            log(curr_time)
            log(self.deadline)

            for act in self.actions:
                log("Current Action is")
                log(act)

                next_st1_l = curr_st_l.copy()
                next_st2_l = curr_st_l.copy()
                next_st1_l[0] = next_st1_l[0] + 1  # increase current time by 1
                next_st2_l[0] = next_st2_l[0] + 1  # increase current time by 1

                index_last_action = 2 * act - 1
                index_pt = 2 * act

                if next_st1_l[index_pt] + 1 < self.max_planning_time:  # add only valid state
                    log("Succ State-1 added")
                    next_st1_l[index_pt] = next_st1_l[index_pt] + 1  # increase planning time for the first state
                    next_st1 = tuple(next_st1_l)  # convert it into a tuple for the states list

                    state_space_set.add(next_st1)  # append to the main set
                    tmp.add(next_st1)  # append to the tmp set for loop
                    # log(curr_st)
                    log(next_st1)

                if next_st2_l[index_last_action] + 1 < self.actions_per_plan:  # add only valid state
                    log("Succ State-2 added")
                    next_st2_l[index_last_action] = next_st2_l[index_last_action] + 1  # next_action
                    next_st2_l[index_pt] = 0  # make the planning time = 0
                    next_st2 = tuple(next_st2_l)  # convert it into a tuple for the states list

                    state_space_set.add(next_st2)  # append to the main list
                    tmp.add(next_st2)  # append to the tmp list for loop
                    # log(curr_st)
                    log(next_st2)

            log("Succ State-3 added")
            next_st3_l = curr_st_l.copy()
            next_st3_l[0] = next_st3_l[0] + 1
            next_st3 = tuple(next_st3_l)
            state_space_set.add(next_st3)
            tmp.add(next_st3)

            # log(curr_st)
            log(next_st3)

        log("The States List with Duplicates has")
        # state_space_list.reverse()
        log(len(state_space_set))
        return state_space_set

    def get_state_id_map(self):
        mapping = defaultdict(count().__next__)
        result = []
        for element in self.state_space:
            result.append(mapping[tuple(element)])

        res = dict(zip(result, self.state_space))
        res = collections.OrderedDict(sorted(res.items()))
        return res

    def get_state_ids(self):
        mapping = defaultdict(count().__next__)
        result = []
        for element in self.state_space:
            result.append(mapping[tuple(element)])

        return result

    def isStatePresent(self, st):
        key = {i for i in self.state_id_map if self.state_id_map[i] == st}
        if len(key) > 0:
            assert (len(key) == 1)
            return True
        return False

    # This function takes the state id and action and returns a list of size 0-2
    # of tuples where every tuple is (next_state_id, prob)
    def step(self, st_id, action):
        assert (action in self.actions)
        assert (st_id in self.state_ids)
        curr_st = self.state_id_map[st_id]
        curr_st_l = list(curr_st)
        res = []
        next_st1_l = curr_st_l.copy()
        next_st2_l = curr_st_l.copy()
        next_st3_l = curr_st_l.copy()
        next_st1_l[0] = next_st1_l[0] + 1  # increase current time by 1
        next_st2_l[0] = next_st2_l[0] + 1  # increase current time by 1
        next_st3_l[0] = next_st3_l[0] + 1  # increase current time by 1

        next_st3 = tuple(next_st3_l)
        key3 = {i for i in self.state_id_map if self.state_id_map[i] == next_st3}

        index_last_action = 2 * action - 1
        index_pt = 2 * action

        last_refined_action = next_st1_l[index_last_action]  # next symbolic action to transition depends on action
        pt_invested = next_st1_l[index_pt]
        t_prob = self.planning_dist[action - 1][last_refined_action][pt_invested]

        next_st1_l[index_pt] = next_st1_l[index_pt] + 1  # increase planning time for the first state
        next_st1 = tuple(next_st1_l)  # convert it into a tuple for the states list
        key1 = {i for i in self.state_id_map if self.state_id_map[i] == next_st1}

        #         if len(key1)>0 :
        #             assert(len(key1)==1)
        #             next_st1_id = int(list(key1)[0])
        #             res.append((next_st1_id,float(1-t_prob)))

        next_st2_l[index_last_action] = next_st2_l[index_last_action] + 1  # next_action
        next_st2_l[index_pt] = 0  # make the planning time = 0
        next_st2 = tuple(next_st2_l)  # convert it into a tuple for the states list
        key2 = {i for i in self.state_id_map if self.state_id_map[i] == next_st2}  # since you are looking only in
        # set of valid states, you dont need to check it again here

        #         if len(key2)>0 :
        #             assert(len(key2)==1)
        #             next_st2_id = int(list(key2)[0])
        #             res.append((next_st2_id,float(t_prob)))

        #         if len(res) == 0:
        #             res.append((st_id,0.00))

        if len(key1) > 0 and len(key2) > 0:
            assert (len(key1) == 1)
            assert (len(key2) == 1)
            next_st1_id = int(list(key1)[0])
            res.append((next_st1_id, float(1 - t_prob)))
            next_st2_id = int(list(key2)[0])
            res.append((next_st2_id, float(t_prob)))
        elif len(key1) > 0:
            next_st1_id = int(list(key1)[0])
            res.append((next_st1_id, float(1.0)))
        elif len(key2) > 0:
            next_st2_id = int(list(key2)[0])
            res.append((next_st2_id, float(1.0)))
        elif len(key3) > 0:
            assert (len(key3) == 1)
            next_st3_id = int(list(key3)[0])
            res.append((next_st3_id, 0.00))

        return res

    def get_id_from_state(self, st):
        key = {i for i in self.state_id_map if self.state_id_map[i] == st}
        if len(key) > 0:
            assert (len(key) == 1)
            st_id = int(list(key)[0])
            return st_id

    def get_state_from_id(self, st_id):
        return self.state_id_map[st_id]

    def get_reward_table(self):
        reward = {}
        for i in range(0, self.num_of_states):
            state = self.get_state_from_id(i)
            state_l = list(state)
            r = 0.0
            for j in range(1, self.num_of_plans + 1):
                plan_id = j - 1
                last_refined_action = state_l[2 * j - 1]
                pt_invested = state_l[2 * j]
                # t_prob =  self.planning_dist[plan_id][last_refined_action][pt_invested]
                # it should not be the last action refined because that doesnot mean the plan is over
                last_action_id = self.actions_per_plan - 1
                t_prob = self.planning_dist[plan_id][last_action_id][pt_invested]
                curr_time = state_l[0]
                if t_prob >= 1.0 and curr_time <= self.deadline and pt_invested == self.planning_times[plan_id][
                    last_action_id]:
                    r = r + 1.0
            reward[i] = r
        return reward


# total_time = 20
# num_of_plans = 2
# actions_per_plan = 1
# max_planning_time = 14
# deadline = 10
# planning_dist = [[[0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.5, 0.75, 0.75, 0.75, 0.99, 0.99]],
#                  [[0, 0.1, 0.2, 0.3, 0.6, 0.8, 0.95, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]]]
# planning_times = np.array([[13], [8]])
# actions = [1, 2]
# env = MetaWorldEnv(total_time, num_of_plans, actions_per_plan, deadline, actions, max_planning_time, planning_dist,
#                    planning_times)
# print("State Space Size ",env.num_of_states)
