from env_file import MetaWorldEnv
import numpy as np
from math import *
import random
import gc
import matplotlib.pyplot as plt
from copy import deepcopy
import time
c = 1.41
random.seed(40)
import tracemalloc

class Node:

    def __init__(self, game, done, parent, curr_state_id, action_index):
        self.child = None
        self.T = 0
        self.N = 0
        self.game = game
        self.curr_state_id = curr_state_id
        self.done = done
        self.parent = parent
        # action index that leads to this node
        self.action_index = action_index

    def getUCBscore(self):

        if self.N == 0:
            return float('inf')

        # We need the parent node of the current node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent

        # We use one of the possible MCTS formula for calculating the node value
        return (self.T / self.N) + c * sqrt(log(top_node.N) / self.N)

    def detach_parent(self):
        # free memory detaching nodes
        del self.parent
        self.parent = None

    def create_child(self):

        '''
        We create one children for each possible action of the game,
        then we apply such action to a copy of the current node enviroment
        and create such child node with proper information returned from the action executed
        '''

        if self.done:
            return
        actions = []
        games = []
        for i in range(self.game.num_of_actions):
            actions.append(i)
            new_game = deepcopy(self.game)
            games.append(new_game)

        child = {}
        current_state = self.curr_state_id
        for action, game in zip(actions, games):
            next_state, reward = game.step2(current_state, game.get_action_from_action_index(action))
            done = game.done(next_state)
            child[action] = Node(game, done, self, next_state, action)

        self.child = child

    def explore(self):

        '''
        The search along the tree is as follows:
        - from the current node, recursively pick the children which maximizes the value according to the MCTS formula
        - when a leaf is reached:
            - if it has never been explored before, do a rollout and update its current value
            - otherwise, expand the node creating its children, pick one child at random, do a rollout and update its value
        - backpropagate the updated statistics up the tree until the root: update both value and visit counts
        '''

        # find a leaf node by choosing nodes with max U.
        current = self
        while current.child:

            child = current.child
            max_U = max(c.getUCBscore() for c in child.values())
            actions = [a for a, c in child.items() if c.getUCBscore() == max_U]
            if len(actions) == 0:
                print("error zero length ", max_U)
            action = random.choice(actions)
            current = child[action]

        # play a random game, or expand if needed

        if current.N < 1:
            current.T = current.T + current.rollout()
        else:
            current.create_child()
            if current.child:
                current = random.choice(current.child)
            current.T = current.T + current.rollout()

        current.N += 1
        # update statistics and backpropagate

        parent = current

        while parent.parent:
            parent = parent.parent
            parent.N += 1
            parent.T = parent.T + current.T
            # print("State : ", parent.game.get_state_from_id(parent.curr_state_id))
            # print("N : ",parent.N )
            # print("T : ",parent.T)

    def rollout(self):

        '''
        The rollout is a random play from a copy of the environment of the current node using random moves.
        This will give us a value for the current node.
        Taken alone, this value is quite random, but, the more rollouts we will do for such node,
        the more accurate the average of the value for such node will be. This is at the core of the MCTS algorithm.
        '''

        if self.done:
            return 0
            # print("Rolling out ", self.game.get_state_from_id(self.curr_state_id))
        v = 0
        done = False
        new_game = deepcopy(self.game)
        current_state = self.curr_state_id
        while not done:
            action = random.choice(new_game.actions)
            # print("Action taken ",action)
            next_state, reward = game.step2(current_state, action)
            done = game.done(next_state)
            # print("Next state ", self.game.get_state_from_id(next_state))
            current_state = next_state
            v = v + reward
            if done:
                break
        return v

    def next(self):

        '''
        Once we have done enough search in the tree, the values contained in it should be statistically accurate.
        We will at some point then ask for the next action to play from the current node, and this is what this function does.
        There may be different ways on how to choose such action, in this implementation the strategy is as follows:
        - pick at random one of the node which has the maximum visit count, as this means that it will have a good value anyway.
        '''

        if self.done:
            raise ValueError("game has ended")

        if not self.child:
            raise ValueError('no children found and game hasn\'t ended')

        child = self.child

        max_N = max(node.N for node in child.values())

        max_children = [c for a, c in child.items() if c.N == max_N]

        if len(max_children) == 0:
            print("error zero length ", max_N)

        max_child = random.choice(max_children)

        return max_child, max_child.action_index


def Policy_Player_MCTS(mytree, max_iterations):
    '''
    Our strategy for using the MCTS is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to MCTS, is the best possible next action
    '''
    MCTS_POLICY_EXPLORE = max_iterations  # MCTS exploring constant: the higher, the more reliable, but slower in execution time
    for i in range(MCTS_POLICY_EXPLORE):
        print("explore")
        mytree.explore()

    next_tree, next_action = mytree.next()
    next_tree.detach_parent()
    return next_tree, next_action


it = np.array([1000], dtype=int)
tracemalloc.start()
for iterations_num in it:
    print("Num of iterations ", iterations_num)
    episodes = 1
    rewards = []
    moving_average = []
    time_taken = 0.0
    reward_e = 0
    num_of_plans = 2
    actions_per_plan = 2
    max_planning_time = 6
    deadline = 7
    actions = [1, 2]
    env = MetaWorldEnv(num_of_plans, actions_per_plan, deadline, actions, max_planning_time)
    for e in range(episodes):
        print(e+1)
        print(tracemalloc.get_traced_memory())
        game = env
        start_state_id = env.get_id_from_state(env.start_state)
        curr_state = start_state_id
        done = False
        new_game = deepcopy(game)
        mytree = Node(new_game, False, 0, start_state_id, 0)
        i = 0
        start_time = time.time()
        p = []
        while True:
            print("oo")
            mytree, action = Policy_Player_MCTS(mytree,iterations_num)
            _, reward = game.step2(mytree.curr_state_id, mytree.game.get_action_from_action_index(action))
            reward_e = reward_e + reward
            if mytree.done:
                _, reward = game.step2(mytree.curr_state_id, mytree.game.get_action_from_action_index(action))
                reward_e = reward_e + reward
                break
            p.append(mytree.game.get_action_from_action_index(action))
        print(tracemalloc.get_traced_memory())
        rewards.append(reward_e)
        moving_average.append(np.mean(rewards))
        time_taken = time_taken + time.time()-start_time
        del(new_game)
        del(mytree)
        del(game)
        gc.collect()

    # plt.plot(rewards)
    # plt.plot(moving_average)
    # plt.show()
    print('moving average: ' + str(np.mean(rewards)))
    print("total time: " + str(time_taken))
    tracemalloc.stop()
