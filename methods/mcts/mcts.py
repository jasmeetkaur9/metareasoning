import random
import itertools
import numpy as np
import math




class Node :
    id_iter = itertools.count()

    def __init__ (self, done, depth, st_id = None):
        self.children = {}
        self.parent = None
        self.Q = 0
        self.N = 0
        if st_id == None :
            self.id = next(Node.id_iter) 
        else :
            self.id = st_id
        self.done = done
        self.depth = depth

    def appendChild(self, action, child):
        self.children[action] = child


class MCTSAgent :
    def __init__(self, env, params):
        self.env  = env
        self.env.curr_state = env.curr_state
        self.actions = self.env.action_space.n
        self.cp = int(params.cp)
        self.lookahead_target = int(params.lookahead_target)
        self.rollout_iterations = int(params.rollout_iterations)
        self.verbose = True if params.verbose == "True" else False
        self.agent_type = 1 if params.env == "MetaWorld-v1" else 0

    def run(self, params, node):
        if self.cp == None:
            self.cp = 10

        if self.lookahead_target == None:
            self.lookahead_target = 200

        root_node = node

        counter = 0
        max_depth = 0
        ix = 0
        while True:
            v = self.act(root_node)
            max_depth = max(v.depth - root_node.depth, max_depth)
            simulation_result = self.random_policy(v)
            self.backpropogation(v, simulation_result, root_node)
            counter += 1
            ix += 1
            if ix > self.rollout_iterations:
                break
        if max_depth < self.lookahead_target:
            self.cp = self.cp - 1
        else:
            self.cp = self.cp + 1
        

        if self.verbose :
            for action, child in sorted(root_node.children.items()):
                print(f"Action :{action}, Q : {child.Q}, N : {child.N}, Q/N : {child.Q/child.N}")

        best_child = max(root_node.children.values(), key=lambda x : x.Q)
        best_child_action = best_child.action
        return (best_child_action, best_child, self.cp)
       

    def act(self, node):
        while not node.done:
            if len(node.children) < self.actions:
                expanded_node = self.expand(node)
                return expanded_node
            else :
                node = self.best_child(node, self.cp)
        
        return node

    def expand(self, node):
        remaining_actions = list(filter(lambda action : not action in node.children.keys(), range(self.actions)))
        a = random.choice(remaining_actions)

        if self.agent_type == 1:
            next_state, reward, _, done, info = self.env.step_mw(a, node.id)
            child_node = Node(done, node.depth+1, next_state)
        else :
            obs, reward, _, done, info  = self.env.step(a)
            child_node = Node(done, node.depth+1)
        
        child_node.parent = node
        node.appendChild(a, child_node)
        child_node.action = a
        return child_node

    def best_child(self, node, random_select=False):
        c = self.cp
        if random_select :
            child_values = {k : v.Q/v.N + c * math.sqrt(2 * math.log(node.N) / v.N) for (k, v) in node.children.items()}
            max_v = max(child_values.values())
            am = random.choice([k for (k, v) in child_values.items() if v == max_v])
            best_child = node.children[am]
        else :
            am = max(node.children.values(), key = lambda v_ : v_.Q/v_.N + c * math.sqrt(2 * math.log(node.N)/v_.N))
            best_child = am
        return best_child

    def random_policy(self, node):
        done = node.done
        total_reward = 0
        curr_state = node.id
        
        while not done:
            random_action =  random.choice(np.arange(self.actions))
            if self.agent_type == 1:
                next_state, reward, _, done, info = self.env.step_mw(random_action, curr_state)
                curr_state = next_state
    
            else :
                obs, reward, _, done, info  = self.env.step(random_action)
            total_reward += reward

        return total_reward

    def backpropogation(self, node, simulation_result, root_node):
        while not node is root_node.parent:
            node.N += 1
            node.Q = node.Q + simulation_result
            node = node.parent
            


