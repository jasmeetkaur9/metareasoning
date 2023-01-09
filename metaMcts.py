from __future__ import division
import random


class Action():
    def __init__(self, player, x):
        self.player = player
        self.x = x

    def __str__(self):
        return str(self.x)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.player == other.player

    def __hash__(self):
        return hash((self.x, self.player))


class State:
    def __init__(self, env, st_id):
        self.env = env
        self.state = st_id

    def getPossibleActions(self):
        actions = self.env.actions
        return actions

    def takeAction(self, action):
        res, _, _ = self.env.step(self.state, action)
        list1 = list(res.keys())
        list2 = list(res.values())
        curr_id = (random.choices(list1, weights=list2, k=1))[0]
        return State(self.env, curr_id)

    def isTerminal(self):
        state = self.env.get_state_from_id(self.state)
        state_l = list(state)
        for j in range(1, self.env.num_of_plans + 1):
            plan_id = j - 1
            last_refined_action = state_l[3 * j - 2]
            pt_invested = state_l[3 * j - 1]
            exec_time = state_l[3 * j]
            curr_time = state_l[0]
            last_action_id = self.env.actions_per_plan - 1
            if curr_time > self.env.deadline:
                return True
            elif curr_time + exec_time <= self.env.deadline and last_action_id == last_refined_action:
                return True
        return False

    def getReward(self):
        state_ = self.env.get_state_from_id(self.state)
        state_l = list(state_)
        r = 0
        for j in range(1, self.env.num_of_plans + 1):
            plan_id = j - 1
            last_refined_action = state_l[3 * j - 2]
            pt_invested = state_l[3 * j - 1]
            exec_time = state_l[3 * j]
            last_action_id = self.env.actions_per_plan - 1

            t_prob = self.env.planning_dist[plan_id][last_action_id][pt_invested]
            curr_time = state_l[0]

            if curr_time + exec_time <= self.env.deadline and last_refined_action == last_action_id:
                r = r + 1.0

        return r
