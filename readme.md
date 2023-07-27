# Lunar Lander Project - Deep Q-Network (DQN) Implementation

<img src="img/lunar_lander.gif"/>

## Table of Contents

- [Introduction](#introduction)
- [Q Learning Fundamentals](#q-learning-fundamentals)
- [Deep Q-Network (DQN)](#deep-q-network-dqn)
- [Environment](#environment)
- [Project Goal](#project-goal)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Hyperparameter Analysis](#hyperparameter-analysis)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

The Lunar Lander Project is an implementation of the Deep Q-Network (DQN) algorithm to solve the Lunar Lander v2 environment from OpenAI Gym. This project aims to train an agent using Reinforcement Learning (RL) techniques to successfully land a lunar lander on the moon's surface while avoiding crashes and using minimal fuel. Please check more details in the final report.

## Q Learning Fundamentals

Q-Learning is a popular model-free RL algorithm that allows an agent to learn an optimal action-selection policy based on the state-action value function (Q-function). It employs a Q-table to represent the expected cumulative reward when taking a specific action in a particular state. However, reinforcement learning is unstable when a nonlinear function approximation (like ANN) is used due to the correlations in the sequence of observations and correlations between the action-values and the target values. This is where Deep Q-Networks come into play.

## Deep Q-Network (DQN)

DQN, proposed by Minh et al. [1], is an extension of Q-learning that employs deep neural networks to approximate the Q-function. This neural network, also known as the Q-network, takes the state as input and outputs the Q-values for all possible actions. The DQN algorithm uses experience replay and a target network to stabilize and improve the learning process.

## Environment

The Lunar Lander v2 environment from OpenAI Gym provides a 2D simulation of a lunar lander with various actions such as firing the main engine or side thrusters to control the spacecraft's movement. The agent receives rewards based on its actions, with positive rewards for successful landings and penalties for crashes or fuel consumption.

## Project Goal

The main objective of this project was to train an agent using DQN to successfully land the lunar lander with at least 200 points over 100 consecutive runs. The focus was on the implementatioan of DQN algorithm and exploring the effect of different hyperparameters of the DQN algorithm on the agent's learning and performance.

## Implementation Details

The implementation of the DQN algorithm was done using Python and popular libraries such as TensorFlow and OpenAI Gym. The following steps were undertaken in the implementation:

1. DQN Architecture: A deep neural network was designed as the Q-network, with fully connected layers to approximate the Q-values for each action given a state.
2. Experience Replay: Experience replay was implemented to store agent experiences (state, action, reward, next state) in a memory buffer, and random batches were sampled during training to break temporal correlations. Another advantage of experience replay is that each step of experience is used in many weight updates which offers better data efficiency.
3. Target Network: To enhance stability during training, a separate target network was introduced. It was periodically updated to match the Q-network and reduce the correlations with the target.
4. Exploration vs. Exploitation: An epsilon-greedy policy was adopted to balance exploration and exploitation during the agent's action selection.

## Results

The trained DQN agent demonstrated successful landings in the Lunar Lander environment. It was able to achieve an average of 200 points over 100 consecutive runs, showcasing the effectiveness of the DQN algorithm in solving the lunar landing task.

## Hyperparameter Analysis

In this project, we investigated the impact of various hyperparameters on the performance and convergence of the DQN algorithm. The hyperparameters explored include:

- Learning Rate: The learning rate for updating the neural network's weights during training.
- Epsilon (Exploration Rate): The probability of choosing a random action over the one suggested by the policy.
- Discount Factor (Gamma): The discount factor for future rewards in the Q-learning update equation.
- Experience Replay Buffer Size: The size of the memory buffer used for experience replay.
- Neural Network Architecture: The depth and width of the neural network used as the Q-network.

## Conclusion

The implementation of the Deep Q-Network algorithm successfully solved the Lunar Lander v2 environment from OpenAI Gym, achieving the goal of landing the lunar lander with at least 200 points over 100 consecutive runs. Through the hyperparameter analysis, valuable insights were gained into the sensitivity of the DQN algorithm to different parameters, which can be applied to future RL projects.

The Lunar Lander project serves as a stepping stone for understanding and applying advanced RL techniques, and it can be extended further with additional algorithms or improvements to achieve even better performance.

## References

[1] Minh, V., et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

The code was implemented with Jupyter Notebook. The .py files were attached here. batch_size.py, decay_rates.py, gamma.py, lr.py, and neural.py are used for training model with different hyperparameters. The optimal model was the one in gamma.py with gamma=0.999.

The LunarLander.ipynb is used to test the training model. The optimal model was loaded and tested.

- models folder: store all the training models.
- jupyter notebook folder: store all jupyter notebooks.
- scores folder: store the score and epsilon of each training episode with different hyperparameters.
