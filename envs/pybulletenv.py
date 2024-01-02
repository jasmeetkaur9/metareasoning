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
from envs.metaenv import MetaWorldEnv
import envs



class PR2Env(MetaWorldEnv):

    def __init__(self, params):
        super(PR2Env, self).__init__(params)
        self.PR2 = None
        self.mode = 'P'
    
    def set_pybullet_ob(self, pr2):
        self.PR2 = pr2

    def reset(self):
        self.PR2.reset()
        return super(PR2Env, self).reset()

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

        res = self.transition_function_pr2(st_id, action)

        if self.verbose :
            print("RESULTING NEXT STATES :", res)

        ns = list(res.keys())
        prob = list(res.values())
        curr_id = (random.choices(ns, weights = prob, k = 1))[0]

        reward = self.reward_function(curr_id)
        done = True if self.done(curr_id) else False
        curr_state = self.observation_space.get_state_from_id(curr_id)

        return curr_state, curr_id, reward, {}, done, {}

    def transition_function_pr2(self, st_id, action):   
        state = self.observation_space.get_state_from_id(st_id)
        curr_st_id = st_id 
        state_id_list = [] 
        transition_prob = []
        curr_st_l = list(state)
        curr_time = curr_st_l[0]

        next_st_l = curr_st_l.copy()
        next_st_l[0] = curr_st_l[0] + 1

        index_last_action = 2 * (action+1) - 1
        index_pt = 2*(action + 1)

        if curr_time > self.deadline:
            state_id_list.append(curr_st_id)
            transition_prob.append(1.0)
            return dict(zip(state_id_list, transition_prob))
        

        last_refined_action = curr_st_l[index_last_action]
        pt_invested = curr_st_l[index_pt]
        res, mode, exec_time = self.PR2.step(action, last_refined_action) # now you know which plan to refine
        """
         res == 0, mode == P 
         res == 1, mode == E->P
         res == 2, mode == E last action 
        """
        
        if res == 0:
            next_st_l[index_pt] = next_st_l[index_pt] + 1
            next_st = tuple(next_st_l)
            assert (next_st_l[0] == curr_st_l[0] + 1)
            if not self.observation_space.contains(next_st):
                state_id = self.observation_space.add_state(next_st)
            else :
                state_id = self.observation_space.get_state_id(next_st)
        
        elif res == 1:
            if next_st_l[index_last_action] + 1 < self.actions_per_plan:
                if self.is_done[action][last_refined_action] == 0:
                    self.execution_time[action] = self.execution_time[action] + exec_time
                    self.is_done[action][last_refined_action] = 1
            next_st_l[index_pt] = 0
            next_st_l[index_last_action] = next_st_l[index_last_action] + 1
            next_st = tuple(next_st_l)
            assert (next_st_l[0] == curr_st_l[0] + 1)
            if not self.observation_space.contains(next_st):
                state_id = self.observation_space.add_state(next_st)
            else :
                state_id = self.observation_space.get_state_id(next_st)
        else :
            if self.is_done[action][last_refined_action] == 0:
                    self.execution_time[action] = self.execution_time[action] + self.execution_input[action][last_refined_action]
                    self.is_done[action][last_refined_action] = 1
            next_st = tuple(next_st_l)
            assert (next_st_l[0] == curr_st_l[0] + 1)
            if not self.observation_space.contains(next_st):
                state_id = self.observation_space.add_state(next_st)
            else :
                state_id = self.observation_space.get_state_id(next_st)

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
            elif exec_done and curr_time + exec_time  <= self.deadline and last_action_id == last_refined_action:
                return True
        
        return False

    """ Returns True if State is terminal.
        Arguments :
            State ID
        Returns : Bool
    """
    def done(self, st_id):
        return self.is_terminal(st_id)

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
            if exec_done and curr_time + exec_time  <= self.deadline and last_action_id == last_refined_action:
                reward = reward + 100
        
        return reward
