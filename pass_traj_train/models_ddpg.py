import gym
from gym_torcs import TorcsEnv
import numpy as np
import math
import matplotlib.pyplot as plt
import queue
import random
from collections import deque
import time
from replay_buffer_ddpg import replayBuffer
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser(description = "ActorCritic")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--tau", type=float, default=0.001)
parser.add_argument("--actor_learning_rate", type=float, default=0.0003)
parser.add_argument("--critic_learning_rate", type=float, default=0.0003)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--buffer_size", type=float, default= 10000)

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()

class Actor(nn.Module):
    
        def __init__(self, state_dim = 29, action_dim = 3, hidden_size_one = 600, hidden_size_two = 300):
        
            super(Actor, self).__init__()
            self.input_size = state_dim;
            self.hidden_size_one = hidden_size_one;
            self.hidden_size_two = hidden_size_two;
            self.output_size = action_dim #steering,acceleration, brake
        
            self.l1 = nn.Linear(self.input_size, self.hidden_size_one, bias = False)
            self.l2 = nn.Linear(self.hidden_size_one, self.hidden_size_two, bias = False)
            self.action_layer = nn.Linear(self.hidden_size_two, 3, bias = False)
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            
            self.steer_layer = nn.Linear(self.hidden_size_two, 1, bias = False)
            self.acceleration_layer = nn.Linear(self.hidden_size_two, 1, bias = False)
            self.brake_layer = nn.Linear(self.hidden_size_two, 1, bias = False)
        
            
            self.steer = torch.nn.Sequential(
                                    self.l1,
                                    nn.ReLU(),
                                    #nn.Tanh(),
                                    self.l2,
                                    nn.ReLU(),
                                    self.steer_layer,
                                    nn.Tanh()
                                    )
            self.acceleration = torch.nn.Sequential(                                    
                                    self.l1,
                                    nn.ReLU(),
                                    #nn.Tanh(),
                                    self.l2,
                                    nn.ReLU(),
                                    self.acceleration_layer,
                                    nn.Sigmoid()
                                    )

            self.brake = torch.nn.Sequential(
                                    self.l1,
                                    nn.ReLU(),
                                    #nn.Tanh(),
                                    self.l2,
                                    nn.ReLU(),
                                    self.brake_layer,
                                    nn.Sigmoid()
                                    )

            self.steer.apply(self.weights_init_uniform)


        
        # takes in a module and applies the specified weight initialization
        def weights_init_uniform(self, m):
            classname = m.__class__.__name__
            # apply a uniform distribution to the weights and a bias=0
            if classname.find('Linear') != -1:
                m.weight.data.uniform_(-0.003, 0.003)
                #m.bias.data.fill_(0)
    
        def forward (self, state):
            steer_state = self.steer(state)
            acc_state   = self.acceleration(state)
            brake_state = self.brake(state)
            #print("steer size",steer_state.size())
            #print("acceleration size", acc_state.size())
            #print("brake size", brake_state.size())
            action = torch.cat((steer_state, acc_state, brake_state),1)
            #x = self.l1(state)
            #x = F.relu(x)
            #x = self.l2(x)
            #x = F.relu(x)
            #x = self.action_layer(x)
            #print(x,x.size())
            #action = (self.tanh(x[0][0][0]), self.sigmoid(x[0][0][1]), self.sigmoid(x[0][0][2]))
            #print("In forward",action.size()) 
            return action

class Critic(nn.Module):

        def __init__(self, state_dim = 29, action_dim = 3, hidden_size_one = 300, hidden_size_two = 600, hidden_size_three = 600):
        
            super(Critic, self).__init__()
            self.input_size = (state_dim + action_dim);
            self.hidden_size_one = hidden_size_one;
            self.hidden_size_two = hidden_size_two;
            self.output_size = 1

            self.l1 = nn.Linear(state_dim, self.hidden_size_one, bias = False)
            self.l2 = nn.Linear(self.hidden_size_one, self.hidden_size_two, bias = False)
            self.l2actions = nn.Linear(action_dim, self.hidden_size_two, bias = False)
            #add them both and do relu 
            self.l3 = nn.Linear(self.hidden_size_two, self.output_size, bias = False)
            '''
            self.model = torch.nn.Sequential(
                                    self.l1,
                                    nn.ReLU(),
                                    #nn.Tanh(),
                                    self.l2,
                                    nn.ReLU(),
                                    self.l3,
                                    nn.Tanh()
                                    )
            
            self.model.apply(self.weights_init_uniform)
            '''
            self.l1.weight.data.uniform_(-0.0003, 0.0003)
            self.l2.weight.data.uniform_(-0.0003, 0.0003)
            self.l2actions.weight.data.uniform_(-0.0003, 0.0003)
            self.l3.weight.data.uniform_(-0.0003, 0.0003)

        def weights_init_uniform(self, m):
            classname = m.__class__.__name__
            # apply a uniform distribution to the weights and a bias=0
            if classname.find('Linear') != -1:
                m.weight.data.uniform_(-0.0003, 0.0003)
            #m.bias.data.fill_(0)
        
    
        def forward (self, state, action):
        
            #stateAction = torch.cat([state, action], 1)
            #return self.model(stateAction)
            h1 = nn.functional.relu(self.l1(state))
            h2state = self.l2(h1)
            h2actions = self.l2actions(action)
            #h2 = torch.sum([h2state, h2actions])
            h2 = torch.add(h2state, h2actions)
            outputQ = nn.functional.relu(self.l3(h2))
            return outputQ

class DDPGAgent():
    def __init__(self, env, action_dim = 3, state_dim = 29):# actor, critic, actor_target, critic_target):
            self.env = env
            self.action_dim = action_dim
            self.state_dim = state_dim
            self.critic_lr = args.critic_learning_rate
            self.actor_lr = args.actor_learning_rate
            self.gamma = args.gamma
            self.batch_size = args.batch_size
            self.tau = args.tau

            self.actor = Actor()
            self.critic = Critic()
            self.actor_target = Actor()
            self.critic_target = Critic()

	    if(use_gpu):
		self.actor 		= self.actor.cuda()
		self.critic 		= self.critic.cuda()
		self.actor_target 	= self.actor_target.cuda()
		self.critic_target 	= self.critic_target.cuda()

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)

            self.actor_optimizer = optim.Adam(self.actor.parameters())
            self.critic_optimizer = optim.Adam(self.critic.parameters())

            self.replay_buffer = replayBuffer(args.buffer_size)
            self.loss = nn.MSELoss()
        
    def updateActorCritic(self, batch_size):
            states, actions, state_next,rewards, _ = self.replay_buffer.sample(batch_size)
            states = torch.FloatTensor(states, device = device)
            actions = torch.FloatTensor(actions, device = device)
            rewards = torch.FloatTensor(rewards, device = device).reshape([batch_size,1])
            state_next = torch.FloatTensor(state_next, device = device)

            #forward network
            Q_pres = self.critic.forward(states, actions)
            #print("Beforr target computation", state_next.size())
            action_next = self.actor_target.forward(state_next).detach()
            #print("Actio next",action_next.size())
            Q_next = self.critic_target.forward(state_next, action_next.detach()).detach()
            Q_nexttarget = rewards + Q_next * self.gamma

            #loss functions
            criticLoss = self.loss(Q_nexttarget, Q_pres)
            actorLoss = -1 * self.critic.forward(states, self.actor.forward(states)).mean()

            #update the Q paramters which maps states and actions to the Q value
            self.critic_optimizer.zero_grad();
            criticLoss.backward();
            self.critic_optimizer.step();

            #update the policy parameters which updates the states to actions
            self.actor_optimizer.zero_grad();
            actorLoss.backward();
            self.actor_optimizer.step();

            #update the target network weights with the original network weights
            for tar_param, src_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                tar_param.data.copy_(self.tau * src_param.data + (1.0 - self.tau) * tar_param.data)
    
            for tar_param, src_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                tar_param.data.copy_(self.tau * src_param.data + (1.0 - self.tau) * tar_param.data)

    def selectAction(self, state):
            #state = torch.FloatTensor(state)
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
            action = self.actor.forward(state)
            #print("action shape selectAction",action.size(),action)
            action = action.detach().numpy().reshape(3,)
            #print("action is numpy",action)
            return action

