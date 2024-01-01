#!/scratch/cluster/jkaur/anaconda3/envs/planning/bin/python

from copy import deepcopy
import numpy as np
from itertools import count
from collections import defaultdict
import torch
import collections
import time
import random
import operator
import gym
from gym import spaces
from data.dists import Database


DEBUG = False
random.seed(40)








def transform_input(planning_dist, exec_dist) :
    num_of_plans = planning_dist.shape[0]
    num_of_actions = planning_dist.shape[1]
    planning_input = torch.zeros(num_of_plans, num_of_actions)
    execution_input = torch.zeros(num_of_plans, num_of_actions)
    for plan in range(0, num_of_plans):
        for action in range(0, num_of_actions):
            for t in range(0,planning_dist.shape[2]):
                if(abs(planning_dist[plan][action][t] - 1.0) < 0.0001):
                    planning_input[plan][action] = t+1
                    break

    for plan in range(0, num_of_plans):
        for action in range(0, num_of_actions):
                if(abs(planning_input[plan][action] - 0.0) < 0.0001):
                    planning_input[plan][action] = 1000

    for plan in range(0, num_of_plans):
        for action in range(0, num_of_actions):
            for t in range(0,exec_dist.shape[2]):
                if(abs(exec_dist[plan][action][t] - 1.0) < 0.0001):
                    execution_input[plan][action] = t+1
                    break

    for plan in range(0, num_of_plans):
        for action in range(0, num_of_actions):
                if(abs(execution_input[plan][action] - 0.0) < 0.0001):
                    execution_input[plan][action] = 1000

    return planning_input, execution_input



MAX_STATES = 10000


class MetaWorldObservationSpace(gym.spaces.Space):
    def __init__(self, actions):
        super(MetaWorldObservationSpace, self).__init__()
        self.actions = actions
        self.num_of_states = 0
        self.start_state = tuple([0] * (1 + 2 * self.actions))
        self.state_space = set()
        self.state_id_map = {}
        if not self.contains(self.start_state):
            self.add_state(self.start_state)
    
    def contains(self, obs):
        if obs not in self.state_space:
            return False
        else :
            return True

    """ Slow : Access in O(n)
                state_id = {i for i in self.state_id_map if self.state_id_map[i] == state}
                assert(len(list(state_id)) == 1)
                st_id_old = int(list(state_id)[0])

                Fast : Access in O(1)
    """
    def add_state(self, obs):
        self.state_space.add(obs)
        self.state_id_map[self.num_of_states] = tuple(obs)
        state_id = self.num_of_states
        self.num_of_states = self.num_of_states + 1
        return state_id
    
    def get_state_id(self, obs):
        key_list = list(self.state_id_map.keys())
        val_list = list(self.state_id_map.values())
        position = val_list.index(obs)
        state_id = key_list[position]
        st_id = int(state_id)
        return st_id

    def get_state_from_id(self, st_id):
        return self.state_id_map[st_id]

    def dtype(self):
        return gym.spaces.Tuple.dtype
    
    def sample(self):
        idx = random.randint(0, self.num_of_states-1)
        return self.get_state_from_id(idx)

    def print_space(self):
        print(self.state_space)
    
    def reset(self):
        self.num_of_states = 0
        self.start_state = tuple([0] * (1 + 2 * self.actions))
        self.state_space = set()
        self.state_id_map = {}
        if not self.contains(self.start_state):
            st_id = self.add_state(self.start_state)
        return self.start_state, {}




class MetaWorldEnv(gym.Env):

    def __init__(self, params):
        super(MetaWorldEnv, self).__init__()
        self.data = int(params.data_id)
        self.database = Database()
        planning_dist, execution_dist = self.database.get_dist(self.data)
        self.planning_input, self.execution_input = transform_input(planning_dist, execution_dist)
        self.num_of_plans = int(params.num_symbolic_plans)
        self.actions_per_plan = int(params.num_actions_per_plan)
        self.is_done = np.zeros((self.num_of_plans, self.actions_per_plan), dtype = int)
        
        assert(self.planning_input.shape[0] == self.num_of_plans)
        assert(self.planning_input.shape[1] == self.actions_per_plan)
        assert(self.execution_input.shape[0] == self.num_of_plans)
        assert(self.execution_input.shape[1] == self.actions_per_plan)

        self.deadline = int(params.deadline)
        self.verbose = True if params.verbose == "True" else False
        self.version1 = True if params.version1 == "True"  else False
        self.actions = np.arange(self.num_of_plans)

        self.action_space = spaces.Discrete(self.num_of_plans)
        self.observation_space = MetaWorldObservationSpace(self.num_of_plans)

    def reset(self):
        self.execution_time = np.zeros(self.num_of_plans)
        obs, _ = self.observation_space.reset()
        self.curr_state = obs
        return obs, {}

    """ TODO : This was added only for comptatibility with gym.
        Use step_mw instead.
        Always returns from Start State.
    """
    def step(self, action):
        st_id = self.observation_space.get_state_id(self.curr_state)
        assert (action in self.actions)
        assert (st_id in range(0, self.observation_space.num_of_states))

        res = {}
        ns = {}
        prob = []
        curr_id = st_id
        reward = 0.0

        res = self.transition_function(st_id, action)


        ns = list(res.keys())
        prob = list(res.values())
        curr_id = (random.choices(ns, weights = prob, k = 1))[0]
        curr_state = self.observation_space.get_state_from_id(curr_id)

        reward = self.reward_function(curr_id)
        done = True if self.done(curr_id) else False
        return curr_state, reward, {}, done, {}

    def step_mw(self, action, state):
        st_id = self.observation_space.get_state_id(state)
        assert (action in self.actions)
        assert (st_id in range(0, self.observation_space.num_of_states))

        if self.verbose :
            print("STEP")
            print("ACTION is :", action, "STATE ID :", st_id)

        res = {}
        ns = {}
        prob = []
        curr_id = st_id
        reward = 0.0

        res = self.transition_function(st_id, action)

        if self.verbose :
            print("RESULTING NEXT STATES :", res)

        ns = list(res.keys())
        prob = list(res.values())
        curr_id = (random.choices(ns, weights = prob, k = 1))[0]

        reward = self.reward_function(curr_id)
        done = True if self.done(curr_id) else False
        curr_state = self.observation_space.get_state_from_id(curr_id)

        return curr_state, curr_id, reward, {}, done, {}

    
    def transition_function(self, st_id, action):
        state = self.observation_space.get_state_from_id(st_id)
        curr_st_id = st_id
        if self.verbose :
            print("GENERATING STATES FROM  :", curr_st_id)

        state_id_list = []
        transition_prob = []
        curr_st_l = list(state)
        curr_time = curr_st_l[0]
        if self.verbose :
            print("ACTION :", action)
        
        next_st_l = curr_st_l.copy()
        next_st_l[0] = curr_st_l[0] + 1 

        index_last_action = 2 * (action + 1) - 1
        index_pt = 2 * (action + 1)

        if curr_time > self.deadline:
            state_id_list.append(curr_st_id)
            transition_prob.append(1.0)
            return dict(zip(state_id_list, transition_prob))
        
        last_refined_action = next_st_l[index_last_action] 
        pt_invested = next_st_l[index_pt]
        max_planning = self.planning_input[action][last_refined_action]

        if pt_invested < max_planning :
            next_st_l[index_pt] = next_st_l[index_pt] + 1 # increase planning time 
            next_st = tuple(next_st_l)
            assert (next_st_l[0] == curr_st_l[0] + 1)
            if not self.observation_space.contains(next_st):
                state_id = self.observation_space.add_state(next_st)
            else :
                state_id = self.observation_space.get_state_id(next_st)

        elif pt_invested >= max_planning :
            if next_st_l[index_last_action] + 1 < self.actions_per_plan:
                if self.is_done[action][last_refined_action] == 0:
                    self.execution_time[action] = self.execution_time[action] + self.execution_input[action][last_refined_action]
                    self.is_done[action][last_refined_action] = 1
                next_st_l[index_pt] =  1 # increase planning time 
                next_st_l[index_last_action] = next_st_l[index_last_action] + 1
                next_st = tuple(next_st_l)
                assert (next_st_l[0] == curr_st_l[0] + 1)
                if not self.observation_space.contains(next_st):
                    state_id = self.observation_space.add_state(next_st)
                else :
                    state_id = self.observation_space.get_state_id(next_st)
            else :                   # this is the last action
                if self.is_done[action][last_refined_action] == 0:
                    self.execution_time[action] = self.execution_time[action] + self.execution_input[action][last_refined_action]
                    self.is_done[action][last_refined_action] = 1
                next_st = tuple(next_st_l)
                assert (next_st_l[0] == curr_st_l[0] + 1)
                if not self.observation_space.contains(next_st):
                    state_id = self.observation_space.add_state(next_st)
                else :
                    state_id = self.observation_space.get_state_id(next_st)

        if self.verbose:
            print("ADDING :", curr_st_id, state_id, next_st_l)

        state_id_list.append(state_id) # append the transition 
        transition_prob.append(1.0)
        return dict(zip(state_id_list, transition_prob))


    """ Checks is the state is terminal.
        Arguments :
            State Id
        Returns :
            True if it's terminal
    """
    def is_terminal(self, st_id):
        state = self.observation_space.get_state_from_id(st_id)
        state_l = list(state)
        for j in range(0, self.num_of_plans):
            last_refined_action = state_l[2 * (j+1) - 1]
            pt_invested = state_l[2 * (j+1)]
            curr_time = state_l[0]
            exec_time = self.execution_time[j]
            last_action_id = self.actions_per_plan - 1
            exec_done = self.plan_executed(j)
            if curr_time > self.deadline:
                return True
            elif pt_invested >= self.planning_input[j][last_refined_action] and exec_done and curr_time + exec_time <= self.deadline and last_action_id == last_refined_action:
                return True
        
        return False

    """ Returns True if State is terminal.
        Arguments :
            State ID
        Returns : Bool
    """
    def done(self, st_id):
        return self.is_terminal(st_id)

    def plan_executed(self, plan):
        value = np.prod(self.is_done[plan])
        return (True if value == 1 else False) 


    """ Calculates the Reward of being in a state
        Argument : State Id
        Returns : Reward
    """
    def reward_function(self, st_id):
        reward = 0.00
        state = self.observation_space.get_state_from_id(st_id)
        state_l = list(state)
        for j in range(self.num_of_plans):
            last_refined_action = state_l[2 * (j+1) - 1]
            pt_invested = state_l[2 * (j+1)]
            curr_time = state_l[0]
            exec_time = self.execution_time[j]
            last_action_id = self.actions_per_plan - 1
            exec_done = self.plan_executed(j)
            if pt_invested >= self.planning_input[j][last_refined_action] and exec_done and curr_time + exec_time <= self.deadline and last_action_id == last_refined_action:
                reward = reward + 100
        
        return reward
        









