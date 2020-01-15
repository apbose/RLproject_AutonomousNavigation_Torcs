import numpy as np
import math
import matplotlib.pyplot as plt
import queue
import random
from collections import deque
import time


class replayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size;
        self.buffer = deque(maxlen = buffer_size)

    def push (self, state, action, next_state, reward, done):
        samples = (state, action, next_state, reward, done)
        self.buffer.append(samples)

    def sample (self, batch_size):
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        done_batch = []

        batch_data = random.sample(self.buffer, batch_size)

        for samples in batch_data:
            state, action, next_state, reward, done = samples
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, next_state_batch, reward_batch, done_batch)

    def __len__(self):
        return len(self.buffer)
