[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This is my submission for the Navigation project from Udacity Deep Reinforcement Learning Nanodegree. For this project, I trained an agent to navigate (and collect bananas!) in a large, square world.
I used Double DQN, a modification of the "classic" DQN algorithm, which helps the model to converge faster. Technically this modification is very small, you may see it in `dqn_agent.py` file, in the lines from 95 to 100.

### More details

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download and install the [drlnd repository](https://github.com/udacity/deep-reinforcement-learning). The installation instruction is in their readme file.

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

**Troubleshootting to point 1:** 

In this NanoDegree a very old PyTorch version is used, which requires a corresponding (also prehistoric) version of CUDA. If you have already installed CUDA you may face PyTorch installation problems like this:

`No matching distribution found for torch==0.4.0 (from unityagents==0.4.0)`

If so, just install PyTorch using conda, namely:

`conda install pytorch=0.4.0 -c pytorch`

it will install PyTorch and required non-python software (an older CUDA version). Then run the installation again.

Understanding this took me more time than solving the entire project. I hope that Udacity will upgrade this project soon or at least provide a good, easy to find troubleshooting.


### Instructions

Just run the `Navigation.ipynb` notebook and go through the code. To make it easier, I kept the structure of the code from the DQN exercise, so:
- `Navigation.ipynb` - the notebook with the main code.
- `dqn_agent.py` - the code for the agent
- `model.py` - the code for the torch model (dense NN)
- `checkpoint1.pth` - trained model saved
- `Report.md` - report with the implementation details.
- `Readme.md` - this file
