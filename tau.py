#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import keras

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from datetime import datetime

tf.compat.v1.disable_eager_execution()


# In[7]:


class DQN:
    def __init__(self, env, tau=0.003, gamma=0.99, hidden_size=64, learning_rate=0.0005, batch_size=32):
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.C = 4
        self.memory = deque(maxlen=100000)
        self.count = 0
        
        self.Q, self.target_Q  = self.QNetwork(), self.QNetwork()
    
    def QNetwork(self, state_size=8, action_size=4):
        model = Sequential()
        # we should try different activation function and compare them
        model.add(Dense(self.hidden_size, activation="relu"))
        model.add(Dense(self.hidden_size, activation="relu"))
        model.add(Dense(action_size))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model
    
    def memory_append(self, experience):
        self.memory.append(experience)
    
    def action(self, state, epsilon):
        if np.random.random() >= epsilon:
            return np.argmax(self.Q.predict(state)[0])
        else:
            return self.env.action_space.sample()
    
    def train_weight(self):
        weights = self.Q.get_weights()
        target_weights = self.target_Q.get_weights()
        for i, _ in enumerate(weights):
            target_weights[i] = target_weights[i] + self.tau * (weights[i] - target_weights[i])
        self.target_Q.set_weights(target_weights)
        
    def learn(self):
        self.count += 1
        self.count %= self.C
        if len(self.memory) < self.batch_size:
            return None
        batch_state, batch_target = [], []
        sample_batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in sample_batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.target_Q.predict(next_state)[0])
            target_val = self.Q.predict(state)
            target_val[0][action] = target

            batch_state.append(state[0])
            batch_target.append(target_val[0])
        self.Q.fit(np.array(batch_state), np.array(batch_target), epochs=1, verbose=0)
        if self.count == 0:
            self.train_weight()
    
    def save(self, file):
        self.Q.save(file)


# In[8]:


# Training
def train(gamma=0.99, learning_rate=0.0005, name="test", decay_rate=0.0001, batch_size=32, tau=0.003):
    env = gym.make("LunarLander-v2")
    train_episodes = 2000 
    max_steps = 1000
    epsilon=1.0
    epsilon_stop=0.01
    
    # create a score window to terminate training early
    score = []
    score_moving_window = deque(maxlen=100)
    epsilons = []
    # create the agent
    agent = DQN(env=env, learning_rate=learning_rate, gamma=gamma, batch_size=batch_size, tau=tau)
    
    for episode in range(1, train_episodes):
        total_reward = 0
        step = 0
        state = env.reset().reshape(1, 8)
        epsilons.append(epsilon)
        while step < max_steps:
            step += 1
            
            # use epsilon greedy policy to get action
            action = agent.action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1,8])
            total_reward += reward

            # add the transition to replay memory
            agent.memory_append((state, action, reward, next_state, done))
            
            # learn
            agent.learn()
            state = next_state

            if done:
                break
        
        # record score and score window
        score.append(total_reward)
        score_moving_window.append(total_reward)
        # update epsilon
        epsilon = epsilon_stop + (epsilon - epsilon_stop)*np.exp(-decay_rate*step)
        print('Episode: {}'.format(episode),
              'Total reward: {}'.format(total_reward),
              'Mean score: {}'.format(np.mean(score_moving_window)),
              'Explore P: {}'.format(epsilon),
              datetime.now().time())
        if np.mean(score_moving_window) >= 220:
            agent.save(name)
            print("\n Achieve Mean Score of 220 over last 100 episodes with total {} episodes!".format(episode))
            break
    agent.save(name)
    print(print("\n Not achieve Mean Score of 220 over past 100 episodes with total {} episodes!".format(episode)))
    env.close()
    return (score, epsilons)


# In[9]:


taus = [0.0005, 0.001, 0.003, 0.01, 0.025]
scores, epsilons = [0,0,0,0,0], [0,0,0,0,0]


# In[10]:


for _, tau in enumerate(taus):
    scores[_], epsilons[_] = train(tau=tau, name="tau"+str(tau))
    pd.Series(scores[_]).to_csv("scores_tau"+str(tau))
    pd.Series(epsilons[_]).to_csv("epsilons_tau"+str(tau))


# In[22]:


fig=plt.figure(figsize=(12,6),dpi=200)
ax=fig.add_axes([0,0,1,1])
tau0005=ax.plot(pd.Series(scores[0]).rolling(window=100).mean(),'r',label='tau0.0005')
tau001=ax.plot(pd.Series(scores[1]).rolling(window=100).mean(),'b',label='tau0.001',alpha=1)
tau003=ax.plot(pd.Series(scores[2]).rolling(window=100).mean(),'g',label='tau0.003',alpha=1)
tau01=ax.plot(pd.Series(scores[3]).rolling(window=100).mean(),'y',label='tau0.01',alpha=1)
tau025=ax.plot(pd.Series(scores[4]).rolling(window=100).mean(),'m',label='tau0.025',alpha=1)
ax.legend(loc=0)
ax.set_title("The Effect of tau",size=20)
ax.set_xlabel("Episodes")
ax.set_ylabel("Score")


# In[ ]:




