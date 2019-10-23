import gym
import torch
import torch.nn as nn
from itertools import count
from torch.distributions import Bernoulli
import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from collections import deque
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def select_action(self, state):
        with torch.no_grad():
            prob = self.forward(state)
            b = Bernoulli(prob)
            action = b.sample()
        return action.item()


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


env = gym.make('CartPole-v0')
policy = PolicyNetwork().to(device)
value = ValueNetwork().to(device)
optim = torch.optim.Adam(policy.parameters(), lr=1e-4)
value_optim = torch.optim.Adam(value.parameters(), lr=3e-4)
gamma = 0.99
writer = SummaryWriter('a2c_logs')
memory = Memory(200)
batch_size = 32
is_learn = False
steps = 0

for epoch in count():
    state = env.reset()
    episode_reward = 0

    for time_steps in range(200):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = policy.select_action(state_tensor)
        next_state, reward, done, _ = env.step(int(action))
        episode_reward += reward
        memory.add((state, next_state, action, reward, done))
        if done:
            break
        state = next_state
    experiences = memory.sample(memory.size())
    batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*experiences)
    batch_state = torch.FloatTensor(batch_state).to(device)
    batch_next_state = torch.FloatTensor(batch_next_state).to(device)
    batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
    batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
    batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

    # print(batch_state.shape, batch_next_state.shape, batch_action.shape, batch_reward.shape)

    with torch.no_grad():
        value_target = batch_reward + gamma * (1 - batch_done) * value(batch_next_state)
        advantage = value_target - value(batch_state)
    prob = policy(batch_state)
    # print(prob.shape)
    b = Bernoulli(prob)
    log_prob = b.log_prob(batch_action)
    loss = - log_prob * advantage
    # print(loss.shape)
    loss = loss.mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    writer.add_scalar('action loss', loss.item(), epoch)

    value_loss = F.mse_loss(value_target, value(batch_state))
    value_optim.zero_grad()
    value_loss.backward()
    value_optim.step()
    writer.add_scalar('value loss', value_loss.item(), epoch)

    writer.add_scalar('episode reward', episode_reward, epoch)
    if epoch % 10 == 0:
        print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
        torch.save(policy.state_dict(), 'a2c-policy.para')


