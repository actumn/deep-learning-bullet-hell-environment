import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/game")
from main import Main
import numpy as np
from collections import deque
from skimage import color, transform, exposure
import random

game = Main()
D = deque()
from config import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=(4 - 1) // 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=(2 - 1) // 2)

        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))




# Obtain the starting state
r_0, s_t, s_f = game.MainLoop(3)
# Failsafe press of x - sometimes startup lags affects ability to enter the game successfully
# pyautogui.press('x')
# Turn our screenshot to gray scale, resize to num_of_cols*num_of_rows, and make pixels in 0-255 range
s_t = color.rgb2gray(s_t)
s_t = transform.resize(s_t, (num_of_cols, num_of_rows))
s_t = exposure.rescale_intensity(s_t, out_range=(0, 255))
s_t = np.stack((s_t, s_t, s_t, s_t), axis=2)
# In Keras, need to reshape
s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*num_of_cols*num_of_rows*4

t = 0

while True:
    # pyautogui.keyDown('x')
    # pyautogui.keyUp('left')
    # pyautogui.keyUp('right')
    explored = False

    loss = 0  # initialize the loss of the network
    Q_sa = 0  # initialize state
    action_index = 0  # initialize action index
    r_t = 0  # initialize reward
    a_t = np.zeros([num_of_actions])  # initalize acctions as an array that holds one array [0, 0]

    # choose an action epsilon greedy, or the action that will return the highest reward using our network
    # i chose to create an arbitrary policy before it starts learning to try and explore as much as it can
    if t < observe:
        action_index = 1 if random.random() < 0.5 else 0
    else:
        action_index = random.randint(0, num_of_actions - 1)  # choose a random action
        explored = True
    # pyautogui.keyDown(action_array[action_index])
    # keyboard.press(action_array[action_index])

    # execute the action and observe the reward and the state transitioned to as a result of our action

    r_t, s_t1, terminal = game.MainLoop(action_index)

    # get and preprocess our transitioned state
    print(s_t1.shape)
    s_t1 = color.rgb2gray(s_t1)
    print(s_t1.shape)
    s_t1 = transform.resize(s_t1, (num_of_rows, num_of_cols))
    print(s_t1.shape)
    s_t1 = exposure.rescale_intensity(s_t1, out_range=(0, 255))
    print(s_t1.shape)

    s_t1 = s_t1.reshape(1, s_t1.shape[0], s_t1.shape[1], 1)  # 1x80x80x1
    print(s_t1.shape)
    s_t1 = np.append(s_t1, s_t[:, :, :, :3], axis=3)
    print(s_t1.shape)
