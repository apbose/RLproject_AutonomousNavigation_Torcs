# RLproject_AutonomousNavigation_Torcs

Rhis project aims to train an agent
in the TORCs simulator environment using RL algorithms.
Starting with naive algorithm using only available state space
variables, Deep Deterministic Policy Gradients (DDPG) is
trained. We further implement Behavior Cloning (BC) and our
variation of Generative Adversarial Imitation Learning (GAIL).
In GAIL, we train a multi-modal based deep RL algorithm
including visual inputs and causal information encoding to
train the agent. We also present the utilization of the causality
of the latent code in an expert trajectory consisting of different
sub-tasks and leverage this in the model

The code is developed based on the paper https://arxiv.org/abs/1703.08840 
The detailed explanation of TORCS for AI research . (https://arxiv.org/pdf/1304.1672.pdf)

# Requirements

Python 2.7<br />
Tensorflow 0.12.1<br />
PyTorch 0.4.1<br />
Keras 1.2.2<br />
xautomation<br />

# Run Instructions

Install all the dependencies for TORCs ( http://torcs.sourceforge.net/ for more info)<br />
git clone https://github.com/apbose/RLProject_AutomousNavigation_Torcs<br />
For the two subtasks <br />

pass -<br /> 
cd pass_traj_train<br />
Training - <br />
Latent code embedded model - python train.py<br />
Behavior Cloning model - python BC.py<br />
DDPG Model - python ddpgMain.py<br />
Evaluation - python evaluate.py<br />

turn -<br />
cd turn_traj_train<br />
Training - <br />
Latent code embedded model - python train.py<br />
Behavior Cloning model - python BC.py<br />
DDPG Model - python ddpgMain.py<br />
Evaluation - python evaluate.py<br />

For the causal model implementation
cd pass_traj_train<br />
Training - python train.py -causal 1<br />
Evaluation - python evaluate.py<br />




