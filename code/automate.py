from uiautomator import Device
import numpy as np
import time
import timeit
import urllib3

import torchvision.transforms as T
from PIL import Image
from scipy import misc

import pandas as pd

import torch

from collections import namedtuple

import sys
import os
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T

from subprocess import call

from torch.autograd import Variable

FNULL = open(os.devnull, 'w')
APP_UNDER_TEST_ROOT = "/Users/vini/Dev/uni/dissertation/code/sample_app/"

http_client = urllib3.PoolManager()

# call("adb shell settings put global window_animation_scale 0.0", shell=True)
# call("adb shell settings put global transition_animation_scale 0.0", shell=True)
# call("adb shell settings put global animator_duration_scale 0.0", shell=True)


compile_reporter = "javac -cp lib/org.jacoco.ant-0.7.9-nodeps.jar:. ReportGenerator.java"


class Action:

    def __init__(self, gui_object, action_type='click'):
        self.gui_object = gui_object
        self.action_type = action_type

    def execute(self):
        if self.action_type == 'click':
            self.gui_object.click()

resize = T.Compose([T.ToPILImage(), T.Scale(38, interpolation=Image.CUBIC), T.ToTensor()])

class AndroidEnv:

    def __init__(self, app_package, screen_size):
        self.app_package = app_package
        self.device = Device()
        self.screen_size = screen_size
        self.complete = misc.imread("complete.png")

    def reset(self):
        self._exec("adb forward tcp:8981 tcp:8981")
        self._exec(f"adb shell am force-stop {self.app_package}")
        self._exec(f"adb shell pm clear {self.app_package}")
        self._exec(f"adb shell monkey -p {self.app_package} 1")
        return self._get_screen(), self._get_actions()

    def step(self, action):
        action.execute()
        obs = self._get_screen()

        img = misc.imread("state.png")
        done = np.array_equal(img[100:], self.complete[100:])
        reward = 1000 if done else 0
        return obs, self._get_actions(), reward, done

    def _exec(self, command):
        call(command, shell=True, stdout=FNULL)

    def _get_actions(self):
        actions = []
        for gui_obj in self.device():
            if gui_obj.clickable:
                actions.append(Action(gui_obj))
        return actions

    def _get_screen(self):
        self.device.screenshot("state.png")
        img = misc.imread("state.png")
        return self._image_to_torch(img)

    def _image_to_torch(self, image):
        # img_resized = misc.imresize(image, size=0.1)
        screen_transposed = image.transpose((2, 0, 1))
        screen_scaled = np.ascontiguousarray(screen_transposed, dtype=np.float32) / 255
        torch_img = torch.from_numpy(screen_scaled)
        return resize(torch_img).unsqueeze(0)

    def _get_current_coverage(self):
        start_time = timeit.default_timer()
        # write_report_cmd = "adb shell am broadcast -a rl.example.com.myapplication.intent.action.WRITE_REPORT"
        # read_report_cmd = f"adb pull /sdcard/coverage.exec {APP_UNDER_TEST_ROOT}app/build/outputs/code-coverage/coverage.exec"

        with http_client.request("GET", "http://localhost:8981", preload_content=False) as r, open("coverage/coverage.exec", "wb") as coverage_file:
            coverage_file.write(r.read())
        generate_report_cmd = f"java -cp lib/org.jacoco.ant-0.7.9-nodeps.jar:. ReportGenerator {APP_UNDER_TEST_ROOT}"
        self._exec(generate_report_cmd)
        # self._exec(f"{write_report_cmd} && {read_report_cmd} && {generate_report_cmd}")
        df = pd.read_csv("coverage/report.csv")
        missed, covered = df[['LINE_MISSED', 'LINE_COVERED']].sum()
        print(f"Complete in {timeit.default_timer() - start_time} seconds")
        return covered / (missed + covered)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(320, 30)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
dtype = torch.FloatTensor

model = DQN()
memory = ReplayMemory(10000)
optimizer = optim.RMSprop(model.parameters())
model.type(dtype)

steps_done = 0
def select_action(state, actions):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        vals = model(Variable(state.type(dtype), volatile=True)).data[0]
        max_idx = vals[:len(actions)].max(0)[1][0]
        return torch.LongTensor([[max_idx]])
    else:
        return torch.LongTensor([[random.randrange(len(actions))]])

d = Device()

app_package = "rl.example.com.myapplication"

env = AndroidEnv(app_package, dict(width=1080, height=1920))


print(env._get_current_coverage())

last_sync = 0
def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    # We don't want to backprop through the expected action values and volatile will save us
    # on temporarily changing the model parameters' requires_grad to False!
    non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)
    non_final_next_states = Variable(non_final_next_states_t, volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # if USE_CUDA:
        # state_batch = state_batch.cuda()
        # action_batch = action_batch.cuda()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch).cpu()

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].cpu()
    # Now, we don't want to mess up the loss with a volatile flag, so let's clear it.
    # After this, we'll just end up with a Variable that has requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def run():
    episode_durations = []
    for i_episode in count(1):
        # Initialize the environment and state
        print(f"\n\nStarting epoch {i_episode}")
        state, actions = env.reset()
        for t in count():
            # Select and perform an action
            action = select_action(state, actions)
            next_state, actions, reward, done = env.step(actions[action[0][0]])
            reward = torch.Tensor([reward])

            if done:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                print(f"Epoch complete in {t + 1} steps")
                episode_durations.append(t+1)
                break
