import numpy as np
import gym
from collections import deque
import random


class OUNoise():
    def __init__(self, action_dim, mu = 0, theta = 0.15, max_sigma = 0.3, min_sigma = 0.3, decay_period = 100000):
        self.mu             = mu
        self.theta          = theta
        self.sigma          = max_sigma
        self.max_sigma      = max_sigma
        self.min_sigma      = min_sigma
        self.decay_period   = decay_period
        self.action_dim     = action_dim
        self.low            = np.array([0,0,-1])
        self.high           = np.array([1,1,1])
     

    def reset(self):
        self.state =  np.ones(self.action_dim) * self.mu


    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        #print("state shape",self.state.shape)
        return self.state 

    def get_action(self, action, t = 0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
