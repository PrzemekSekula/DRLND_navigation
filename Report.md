[//]: # (Image References)

[image1]: ./scores.png "Scores"

# Navigation Project - Report

## Overview

- The goal of this project was to train an agent that navigates in a large, square world, collect yellow bananas and avoid blue ones.
- I used Double DQN, a modification of the "classic" DQN algorithm, which helps the model to converge faster. Technically this modification is very small, you may see it in `dqn_agent.py` file, in the lines from 95 to 100.
- It took ~480 episodes to meet the requirements (an average reward (over 100 episodes) of at least +13). However, if you continue training the model is getting better and better. For this submission, I used the model trained for 1,000 episodes, but during the tests, I got even better results while training longer.
- This model uses a Unity environment provided by Udacity. For more information check the `README.md` where the installation process is described.


## Learning Algorithm

#### Double DQN

Double DQN, a modification of "classic" DQN algorithm has been implemented. The original DQN algorithm was published [here](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) whereas the Double DQN modification is published [here](https://arxiv.org/abs/1509.06461). In short, Double DQN helps with overestimation of the max action-value functions.

The DQN an update rule is as follows:
$\Delta w = \alpha ( R + \gamma\;\underset{a}{max}\,\hat{q}(S', a, w) - \hat{q}(S, a, w^-) )\nabla_w \hat{q}(S, a, w)$

Where $w$ are the weights of the local network, $w-$ are the weights of the target network and the $R + \gamma\;\underset{a}{max}\,\hat{q}(S', a, w)$ expression is called TD target .

In in "classic" DQN the TD target can be denoted as

$TD_{target} = R + \gamma\;\hat{q}(S', \underset{a}{arg\,max}\;\hat{q}(S', a, w), w)$

As the same model parameters are used to find the actions that correspond to the highest values of $\hat{q}$ and then to determine the $\hat{q}$, it may lead to overestimation of the $\hat{q}$ function. In order to avoid it, in Double DQN it is recommended to use a different network to determine the highest-value action, and a different one to determine the value that corresponds to this action. As in DQN we already have two networks (local and target network), we are just using the local network to determine the actions with the highest $\hat{q}$ values and then target network to determine the values of $\hat{q}$. So our TD target looks as follows:

$TD_{target} = R + \gamma\;\hat{q}(S', \underset{a}{arg\,max}\;\hat{q}(S', a, w), w^-)$

In the code the difference is only about computing the next actions, in can be find in `dqn_agent.py` file in the lines 95-100. 
For "classic" DQN it is:
<code>
        qmax_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
</code>

Whereas for Double DQN it is:
<code>
        actions_max = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        qmax_next = self.qnetwork_target(next_states).detach().gather(1, actions_max)
</code>

For more information please read the aforementioned papers.

#### Neural Network
A fully connected (dense) neural network was used, with the following parameters:
- Input layer - 37 dimensions
- Hidden layer - 128 neruons, relu activation function
- Hidden layer - 64 neruons, relu activation function
- Hidden layer - 32 neruons, relu activation function
- Output layer - 4 neurons (one neuron for each of four actions)


#### Hyperparameters
No experiments here, I just took the hyperparameters from the DQN exercise, and they worked. All the hyperparameters are defined in `dqn_agent.py` file and in the `Navigation.ipynb` notebook, in `dqn()` function

Parameters from `dqn_agent.py`
<code>
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the networ
</code>

Parameters from `dqn()` function

<code>
n_episodes=1000
max_t=1000, 
eps_start=1.0
eps_end=0.01, 
eps_decay=0.995
</code>

## Average scores
By the end of the training process, the average scores were pushed up to above 16. The model needed around 480 episodes to meet the requirements (score averaged over 100 consecutive episodes is equal to or greater than 13). The rewards are presented in the image below.

![Scores][image1]

## Ideas for future work
- Implement the entire Rainbow algorithm. It may be challenging but it should help me to understand the modifications of DQN better.
- Implement the pixel-input based version of the algorithm (must-do for me :>)
- Modify the given Unity environment. I have a few ideas on how to make it more attractive while keeping it simple.
- Test the impact of the fluctuating exploration/exploitation ratio. I noticed in a couple of environments, that sometimes even if the averaged scores are good, the agent can (in some cases) perform weird actions. Surprisingly, if during the training process the $\epsilon$ value is fluctuating (instead of decreasing). I think I should test it, it is pretty cool.


