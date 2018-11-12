import math
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/game")
from main import Main
import numpy as np
from collections import deque, namedtuple
from skimage import color, transform, exposure
import random

game = Main()
queue = deque()
from config import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=(4 - 1) // 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=(2 - 1) // 2)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        self.head = nn.Linear(10, num_of_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x


def transform_state(single_state):
    # Turn our screenshot to gray scale, resize to num_of_cols*num_of_rows, and make pixels in 0-255 range
    single_state = color.rgb2gray(single_state)
    single_state = transform.resize(single_state, (num_of_cols, num_of_rows))
    single_state = exposure.rescale_intensity(single_state, out_range=(0, 255))
    single_state = torch.from_numpy(single_state)
    single_state = single_state.unsqueeze(0).unsqueeze(0)
    single_state = single_state.to(device, dtype=torch.float)
    return single_state


model = DQN().to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('./weights_pytorch'))
optimizer = optim.Adam(model.parameters())
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'is_terminal'))

# Obtain the starting state
_, curr_state, _ = game.MainLoop(3)
# Fail safe press of x - sometimes startup lags affects ability to enter the game successfully
curr_state = transform_state(curr_state)
curr_state = torch.cat((curr_state, curr_state, curr_state, curr_state), 1)  # 4 image stack

steps = 0
while True:
    loss = 0  # initialize the loss of the network
    action = 0  # initialize action index
    Q_sa = [0]  # initialize state

    # choose an action epsilon greedy, or the action that will return the highest reward using our network
    # i chose to create an arbitrary policy before it starts learning to try and explore as much as it can
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps / eps_decay)
    if random.random() <= eps_threshold:
        action = 1 if random.random() < 0.5 else 0  # choose a random action
    else:
        q_index = model(curr_state).max(1)[1]  # input a stack of 4 images, get the prediction
        action = q_index.item()

    # execute the action and observe the reward and the state transitioned to as a result of our action
    reward, next_state, is_terminal = game.MainLoop(action)

    # get and pre-process our transitioned state
    next_state = transform_state(next_state)
    next_state = torch.cat((next_state, curr_state[:, :3]), 1)  # retain 4 image series

    '''
    We need enough states in our experience replay deque so that we can take a random sample from it of the size we declared.
    Therefore we wait until a certain number and observe the environment until we're ready.
    '''
    if steps > observe:
        # sample a random minibatch of transitions in D (replay memory)
        transitions = random.sample(queue, batch_size)
        batch = Transition(*zip(*transitions))

        # Begin creating the input required for the network:
        # Inputs are our states, outputs/targets are the Q values for those states
        # we have 32 images per batch, images of 128x128 and 4 of each of these images.
        # 32 Q values for these batches
        inputs = torch.cat(batch.state)
        targets = model(inputs)  # 32, 2

        input_next_states = torch.cat(batch.next_state)
        Q_sa = model(input_next_states).max(1)[0]
        for i in range(batch_size):
            if batch.is_terminal[i]:
                targets[i, batch.action[i]] = death_reward
            else:
                targets[i, batch.action[i]] = batch.reward[i] + gamma * Q_sa[i]

        # train the network with the new values calculated with Q-learning and get loss of our network for evaluation
        outputs = model(inputs)
        loss = F.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    queue.append((curr_state, action, reward, next_state, is_terminal))
    if len(queue) > exp_replay_memory:
        queue.popleft()

    curr_state = next_state
    steps += 1

    if steps % timesteps_to_save_weights == 0:
        torch.save(model.state_dict(), './weights_pytorch')

    print("Timestep: %d, Action: %d, Reward: %.2f, Q: %.2f,Loss: %.2f" % (steps, action, reward, Q_sa[-1], loss))