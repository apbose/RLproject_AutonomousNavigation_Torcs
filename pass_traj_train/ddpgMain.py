import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


from gym_torcs import TorcsEnv

from OUNoise_ddpg import OUNoise
from models_ddpg import Actor
from models_ddpg import Critic
from models_ddpg import DDPGAgent
from replay_buffer_ddpg import replayBuffer
from utils import *

parser = argparse.ArgumentParser(description = "Main")
parser.add_argument("--train", type = int, default = 1)
parser.add_argument("--episode_count", type = int, default = 5000)
args = parser.parse_args()




def main():
    #cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    env = TorcsEnv(throttle=True, gear_change=False)


    agent = DDPGAgent(env)      #actor,critic, actorTarget, criticTarget initialised here
    noise = OUNoise(action_dim = 3)
    #print("actionspace",env.action_space)
    batch_size = 128
    rewards = []
    distance = []
    avg_rewards = []
    episodes = []
    path_actor = "experiments/model_actor_ddpg" + str(5000)
    path_actor_target = "experiments/model_actor_target_ddpg" + str(5000)
    path_critic = "experiments/model_critic_ddpg"+ str(5000)
    path_critic_target = "experiments/model_critic_target_ddpg" + str(5000)
    if not args.train:
        try:
            actor.load_state_dict(torch.load(path_actor))
            actor_target.load_state_dict(torch.load(path_actor_target))
            critic.load_state_dict(torch.load(path_critic))
            critic_target.load_state_dict(torch.load(path_critic_target))
            print "Weights laoded successfully"
            time.sleep(2)
        except:
            print "Error in loading weights"
            assert(False)
    file_path = "/home/mathew/Documents/RL/InfoGAIL/wgail_info_0/ddpgOutput.txt" 
    f = open(file_path, "w+") 
    f.close()

    for episode in range(args.episode_count):
	episodes.append(episode)
        print ("Episode", episode, "ReplayBuffer", len(agent.replay_buffer))

        if (np.mod(episode,3) == 0):
            ob = env.reset(relaunch = True)
        else:
            ob = env.reset(relaunch = True)
            
        stateInput = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        #print("obangleshape",ob.angle.shape)
        #print("obtrackshape", ob.track.shape)
        #print("obtrackposhape", ob.trackPos.shape)
        #print("speedxshape", ob.speedX.shape)
        #print("speedyshape", ob.speedY.shape)
        #print("speedzshape", ob.speedZ.shape)
        #print("wheelspinshape", ob.wheelSpinVel.shape)
        #print("rpmshape", ob.rpm.shape)




        noise.reset()
        episode_reward = 0

        for step in range(500):
            action = agent.selectAction(stateInput.reshape(1, stateInput.shape[0]))
            #print("Action shape after select",action)
            #print("State shape",stateInput.shape)
            action = noise.get_action(action, step)
            #print("Action shape", action)
            new_ob, reward, done, _ = env.step(action) 
            new_stateInput = np.hstack((new_ob.angle, new_ob.track, new_ob.trackPos, new_ob.speedX, new_ob.speedY, new_ob.speedZ, new_ob.wheelSpinVel/100.0, ob.rpm))

            agent.replay_buffer.push(stateInput, action, new_stateInput, reward, done)
        
            #update
            if (len(agent.replay_buffer) > batch_size and args.train):
                agent.updateActorCritic(batch_size)        
        
            stateInput = new_stateInput
            episode_reward += reward

            if done:
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
		f = open(file_path, "a+")
		f.write("episode :" + str(episode) + " reward :" + str(np.round(episode_reward, decimals = 2)) + " average _reward :" + str(np.mean(rewards[-10:0])) + "distance"+ str(new_ob.distFromStart)+"\n")
		f.close()
                print("Closing Torcs")
                #env.end()
                break

        
        
        if(episode % 1000 == 0 and episode != 0 and args.train):
            filename_actor = "model_actor_ddpg"+ str(episode)
            filename_actor_target = "model_actor_target_ddpg"+ str(episode)
            filename_critic =  "model_critic_ddpg"+ str(episode)
            filename_critic_target =  "model_critic_target_ddpg"+ str(episode)
            torch.save(self.actor.state_dict(), filename_actor)
            torch.save(self.actor_target.state_dict(), filename_actor_target)
            torch.save(self.critic.state_dict(), filename_critic)
            torch.save(self.critic_target.state_dict(), filename_critic_target)


	distance.append(new_ob.distFromStart)
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
	rewardsArray = np.array(rewards)
	returns = discount(rewardsArray,0.99)
	f = open(file_path, "a+")
	f.write("Returns at episode :" + str(episode) + " is"+ str(episode_reward)+ " with returns :" + np.array_str(returns) +  " average_reward :" + str(avg_rewards)+ " distance"+ str(new_ob.distFromStart)+ "\n")
	#f.write("Returns at episode: {} is returns: {} with avg_rewards: {}\n", episode, returns, avg_rewards)
	f.close()

    
    print("Finished!!!!")
    print(distance)
    plt.plot(episodes, distance)
    #plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


if __name__ == '__main__':
    main()











