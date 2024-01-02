import torch
import gym
import numpy as np
import  torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf 

from methods.ppo.network import Policy
from methods.ppo.network import Value
from methods.ppo.network import Network

from utils.utils import plot_graph



class Memory:
    def __init__(self, batch_size):

        self.obs = []
        self.action  = []
        self.reward = []
        self.next_obs = []
        self.action_prob = []
        self.done = []
        self.count = 0
        self.advantage = []
        self.td_target = torch.FloatTensor([])
        self.batch_size = batch_size

    def add_memory(self, s, a, r, next_s, done, prob):

        self.obs.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.next_obs.append(next_s)
        self.done.append(1-done)
        self.action_prob.append(prob)

    def generate_batches(self):
        batch_size = self.batch_size
        n_states = len(self.obs)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype = np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        return np.array(self.obs), \
               np.array(self.action), \
               np.array(self.reward), \
               np.array(self.next_obs), \
               np.array(self.done), \
               np.array(self.action_prob), \
               batches

    def clear_memory(self):
        self.obs = []
        self.action  = []
        self.reward = []
        self.next_obs = []
        self.action_prob = []
        self.done = []
        self.count = 0
        self.advantage = []
        self.td_target = torch.FloatTensor([])



class PPOAgent:
    def __init__(self, env, params):
        self.env = env
        self.env_name = params.env
        params = dict(params)
        params.pop('env')
        params.pop('logdir')
        for param, val in params.items():
            exec('self.' + param + '=' + str(val))

        self.action_size = self.env.action_space.n
        if (self.env_name == "MetaWorld-v1" or self.env_name == "PR2-v0"):
            self.agent_type = 1
            self.input_size = 5
        else :
            self.agent_type = 0
            self.input_size = 4
        self.policy_network = Policy(action_size = self.action_size, lr = self.learning_rate, k_epoch = self.k_epoch, input_size=self.input_size)
        self.value_network = Value(action_size = self.action_size, lr = self.learning_rate, k_epoch = self.k_epoch, input_size=self.input_size)
        self.network = Network(action_size = self.action_size, lr = self.learning_rate, k_epoch = self.k_epoch, input_size=self.input_size)
        self.memory = Memory(10)
        
    
    def reset(self):
        if self.agent_type == 1:
            curr_state, info = self.env.reset()
            curr_id = self.env.observation_space.get_state_id(curr_state)
            curr = curr_id
            action = self.env.action_space.sample()
            obs, curr_id, reward, _, done, _ = self.env.step_mw(action, curr_state)
        else :
            self.env.reset()
            action = self.env.action_space.sample()
            obs, reward, _, done, _ = self.env.step(action)
        return obs, reward, {}, done, {}

    def train(self):
        episode = 0
        step = 0
        reward_history = []
        avg_reward = []
        loss = []
        solved = False

        while not solved:
            start_step = step
            episode += 1
            episode_length = 0
            
            
            obs, reward, _, done, _ = self.reset()
            curr_obs = obs
            total_episode_reward = 0

            while not solved:
                step += 1
                episode_length += 1

                prob_action = self.policy_network.pi(torch.FloatTensor(curr_obs))
    
                action = torch.distributions.Categorical(prob_action).sample().item()


                if self.agent_type == 1:
                    obs, next_id, reward, _, done, info  = self.env.step_mw(action, curr_obs)
                    next_obs = obs
                    self.memory.add_memory(curr_obs, action, reward, next_obs, done, prob_action[action].item())
                    curr_obs = next_obs
                    curr_id = next_id
                else :
                    obs, reward, _, done, _ = self.env.step(action)
                    next_obs = obs
                    self.memory.add_memory(curr_obs, action, reward, next_obs, done, prob_action[action].item())
                    curr_obs = next_obs
                    reward = -1 if done else reward
                    total_episode_reward += reward

                if done :
                    episode_length = step - start_step
                    reward_history.append(reward)
                    #avg_reward.append(sum(reward_history[-100:])/100.0)
                    avg_reward.append(sum(reward_history)/len(reward_history))
                    loss.append(float(self.network.loss))

                    if episode > 100 or (len(reward_history) > 100 and sum(reward_history[-100:-1]) / 100 >= 195):
                        solved = True
                    
                    self.env.reset()
                    break
                
            if episode % self.update_freq == 0:
                print('episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                          'loss_p: %.4f, loss_v: %.4f, loss : %.4f , lr: %.4f' % (episode, step, episode_length, total_episode_reward, 
                                                    self.policy_network.loss, self.value_network.loss, self.network.loss, 
                                                    self.network.scheduler.get_lr()[0]))
                for _ in range(self.k_epoch):
                    self.update_network()
            
            if episode % self.plot_every == 0:
                plot_graph(avg_reward, "", "Episodes", "Reward", "Navigation", 0)



    def update_network(self):

        obs_, action_, reward_, next_obs_, done_, action_prob_, batches = self.memory.generate_batches()
        advs = np.zeros(len(reward_), dtype = np.float32)

        next_obs_ = torch.tensor(next_obs_, dtype = torch.float32)
        next_obs_v = self.network.v(next_obs_)
        obs_ = torch.tensor(obs_, dtype = torch.float32)
        obs_v = self.network.v(obs_)
        done_ = torch.tensor(done_, dtype = torch.float32)

        td_target = torch.tensor(reward_) + self.gamma * next_obs_v * done_
        delta = td_target - obs_v
        delta = delta.detach().numpy()

        advs = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advs.append(adv)

        # for i in range(len(reward_)-1):
        #     discount = 1
        #     at = 0
        #     for j in range(i, len(reward_)-1):
        #         at = at + discount * (reward_[j] + self.gamma * next_obs_v[j] * (int(done_[j])) - obs_v[j])
        #         discount = discount * self.gamma * self.lmbda
            
        #     advs[i] = at
        advs = torch.tensor(advs)
        
        for batch in batches:
            obs = torch.tensor(obs_[batch], dtype = torch.float)
            next_obs = torch.tensor(next_obs_[batch], dtype = torch.float)
            done = torch.tensor(done_[batch], dtype = torch.float32)
            action_idxs = torch.tensor(action_[batch], dtype = torch.int64)
            reward = torch.tensor(reward_[batch], dtype = torch.float)

            # obs = torch.gather(torch.tensor(obs_), 0, torch.tensor(batches[i]))
            # next_obs = torch.gather(torch.tensor(next_obs_), 0, torch.tensor(batches[i]))
            # done = torch.gather(torch.tensor(done_), 0, torch.tensor(batches[i]))
            # action_idxs = torch.gather(torch.tensor(action_idxs_), 0, torch.tensor(batches[i]))
            # reward = torch.gather(torch.tensor(reward_), 0, torch.tensor(batches[i]))

            # print(obs.shape, next_obs.shape, done.shape, action_idxs.shape, reward.shape)
            
            pi = self.network.pi(obs)
            new_values = pi.shape[0]
            
            new_probs_a = torch.gather(pi, 1, action_idxs.reshape(new_values, 1))
            old_probs_a = torch.tensor(action_prob_[batch], dtype = torch.float)

            ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))
            surr1 = ratio * advs[batch]
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advs[batch]

            pred_v = self.network.v(obs)
            td_target = torch.tensor(reward, dtype = torch.float32) + self.gamma * (self.network.v(next_obs) * done)
            v_loss = 0.5 * (pred_v - td_target).pow(2)

            entropy = torch.distributions.Categorical(pi).entropy()
            entropy = torch.tensor([[e] for e in entropy])
            loss = (-torch.min(surr1, surr2) + self.v_coef * v_loss - self.entropy_coef * entropy).mean()

            self.network.loss = loss
            self.network.zero_grad()
            self.network.loss.backward()
            self.network.optimizer.step()
            self.network.scheduler.step()


    def update_path(self, length):
        obs = self.memory['obs'][-length:]
        reward = self.memory['reward'][-length:]
        next_obs = self.memory['next_obs'][-length:]
        done = self.memory['done'][-length:]

        td_target = torch.FloatTensor(reward) + \
                    self.gamma * (self.network.v(torch.FloatTensor(next_obs)) * torch.FloatTensor(done))
        
        delta = td_target - self.network.v(torch.FloatTensor(obs))
        delta = delta.detach().numpy()

        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        
        advantages.reverse()

        if self.memory['td_target'].shape == torch.Size([1, 0]):
            self.memory['td_target'] = td_target.data
        else :
            self.memory['td_target'] = torch.cat((self.memory['td_target'], td_target.data), dim = 0)
        
        self.memory['advantage'] = self.memory['advantage'] + advantages




                  







