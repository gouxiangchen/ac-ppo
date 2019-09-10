import gym
import torch
import torch.nn as nn
from itertools import count
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc_mu = nn.Linear(256, 1)
        self.fc_std = nn.Linear(256, 1)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = 2 * self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x)) + 1e-3
        return mu, std

    def select_action(self, state):
        with torch.no_grad():
            mu, std = self.forward(state)
            n = Normal(mu, std)
            action = n.sample()
        return np.clip(action.item(), -2., 2.)


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3, 64)
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


env = gym.make('Pendulum-v0')
policy = PolicyNetwork().cuda()
value = ValueNetwork().cuda()
optim = torch.optim.Adam(policy.parameters(), lr=1e-5)
value_optim = torch.optim.Adam(value.parameters(), lr=3e-5)
gamma = 0.99
writer = SummaryWriter('a2c_logs_pen')
memory = Memory(32)
batch_size = 32
is_learn = False
steps = 0


for epoch in count():
    state = env.reset()
    episode_reward = 0
    k = 0
    for time_steps in range(200):
        k += 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()
        action = policy.select_action(state_tensor)
        next_state, reward, done, _ = env.step([action])
        episode_reward += reward
        reward = (reward + 8.1) / 8.1
        memory.add((state, next_state, action, reward, done))
        if k == 32 or time_steps == 199:
            k = 0
            steps += 1
            experiences = memory.sample(batch_size)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*experiences)
            batch_state = torch.FloatTensor(batch_state).cuda()
            batch_next_state = torch.FloatTensor(batch_next_state).cuda()
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).cuda()
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).cuda()
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).cuda()

            # print(batch_state.shape, batch_next_state.shape, batch_action.shape, batch_reward.shape)

            with torch.no_grad():
                value_target = batch_reward + gamma * (1 - batch_done) * value(batch_next_state)
                advantage = value_target - value(batch_state)
            mu, std = policy(batch_state)
            # print(prob.shape)
            n = Normal(mu, std)
            log_prob = n.log_prob(batch_action)
            loss = - log_prob * advantage
            # print(loss.shape)
            loss = loss.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            writer.add_scalar('action loss', loss.item(), steps)

            value_loss = F.mse_loss(value_target, value(batch_state))
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()
            writer.add_scalar('value loss', value_loss.item(), steps)

            memory.clear()
        if done:
            break
        state = next_state
    writer.add_scalar('episode reward', episode_reward, epoch)
    if epoch % 10 == 0:
        print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
        torch.save(policy.state_dict(), 'a2c-pen-policy.para')




