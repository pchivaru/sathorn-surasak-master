import random
import os
import re
import numpy as np
from networks import DQNNet
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from shared import DISCOUNT, BATCH_SIZE, UPDATE_RATE, EXPLORE, EPSILON


class Agent(object):

    def __init__(self, obs_shape, num_actions, action_method):
        self.num_actions = num_actions

        self.trainer = DQNNet(obs_shape, self.num_actions, learning_rate=0.001)
        self.target = DQNNet(obs_shape, self.num_actions, learning_rate=0.001)

        self.replay = ReplayBuffer(100000)# ALPHA
        self.beta = 95 #???

        self.prev_state = None
        self.prev_action = None
        self.last_change = 0

        self.action_method = action_method

        self.ep = 1
        if self.action_method == 'egreedy_exp':
            self.explore = np.e**(-EXPLORE*(self.ep-1))
        elif self.action_method == 'egreedy_const':
            self.explore = EPSILON

    def load(self, env_id):
        self.trainer.load('Data'+env_id+os.sep+'policy.net')
        self.target.model.set_weights(self.trainer.model.get_weights())

        self.replay.load('Data'+env_id+os.sep+'replay.buffer')
        
        results = re.search(r'(\d+)-(\d+)', env_id)
        eps_0 = int(results[2]) 
        self.ep = eps_0+2

        if self.action_method == 'egreedy_exp':
            self.explore = np.e**(-EXPLORE*(self.ep-1))

        if self.ep >= 20:
            self.explore = 0

        print('loaded from ' + env_id)

    def episode_end(self, env_id):
        self.trainer.save('Data'+env_id+os.sep+'policy.net')
        self.replay.save('Data'+env_id+os.sep+'replay.buffer')
        self.prev_action = None
        self.prev_state = None
        self.last_change = 0
        self.ep += 1
        if self.action_method == 'egreedy_exp':
            self.explore = np.e**(-EXPLORE*(self.ep-1))
        print('New epsilon:', self.explore)
        if self.ep >= 20:
            self.explore = 0

    def act(self, state, reward, done):
        #print(self.ep)
        if self.prev_action is not None:
            self.replay.add(self.prev_state, self.prev_action, reward, state, done)
            if len(self.replay) > BATCH_SIZE * 2: 
                self.train()
            self.last_change += 1
            if self.last_change == UPDATE_RATE:
                self.target.model.set_weights(self.trainer.model.get_weights())
                self.last_change = 0

        if random.uniform(0, 1) < self.explore:
            action = random.choice(range(self.num_actions))
            print('Random action', action)
        else:            
            action = self.trainer.best_action(state)
            print('Best action', action)

        self.prev_state = state
        self.prev_action = action
        return action

    def train(self):
        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)#, self.beta)
        # Discount intermediate no-op state rewards
        future_rewards = self.target.best_value(next_states)
        total_reward = []
        for i in range(len(rewards)):
            actual_reward = 0
            for j, reward in enumerate(rewards[i]):
                actual_reward += (DISCOUNT ** j) * reward
            actual_reward += (DISCOUNT ** len(rewards[i])) * future_rewards[i]
            total_reward.append(actual_reward)

        #self.beta = min(self.beta + 0.000002, 1.0)

        # DDQN
        #next_acts = self.trainer.best_actions(next_states)
        #exp_fut_rew = self.target.get_actions_values(next_states, next_acts)
        #total_reward = rewards + DISCOUNT * self.target.best_value(next_states)#exp_fut_rew

        self.trainer.train(states, actions, total_reward)

        # PER
        #exp_fut_rew = self.trainer.get_actions_values(next_states, next_acts)
        #new_total = rewards + DISCOUNT * exp_fut_rew
        #new_priorities = np.abs(total_reward-new_total) + EPS
        #self.replay.update_priorities(batch_idxes, new_priorities)
