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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
policy = PolicyNetwork().to(device)
old_policy = PolicyNetwork().to(device)
optim = torch.optim.Adam(policy.parameters(), lr=1e-5)
value = ValueNetwork().to(device)
value_optim = torch.optim.Adam(value.parameters(), lr=2e-5)
gamma = 0.9
steps = 0

is_learn = False
writer = SummaryWriter('ppo_logs')


for epoch in count():
    state = env.reset()
    episode_reward = 0
    rewards = []
    states = []
    actions = []
    for time_steps in range(200):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = policy.select_action(state_tensor)
        # print('action : ', action)
        next_state, reward, done, _ = env.step([action])
        episode_reward += reward
        reward = (reward + 8.1) / 8.1

        rewards.append(reward)
        states.append(state)
        actions.append(action)

        state = next_state

        if (time_steps+1) % 32 == 0 or time_steps == 199:
            old_policy.load_state_dict(policy.state_dict())
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                R = value(next_state_tensor)
            for i in reversed(range(len(rewards))):
                R = gamma * R + rewards[i]
                rewards[i] = R
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            for K in range(10):
                steps += 1
                state_tensor = torch.FloatTensor(states).to(device)
                action_tensor = torch.FloatTensor(actions).unsqueeze(1).to(device)
                with torch.no_grad():
                    advantage = rewards_tensor - value(state_tensor)

                with torch.no_grad():
                    old_mu, old_std = old_policy(state_tensor)
                    old_n = Normal(old_mu, old_std)
                # print(value_target.shape, advantage.shape)
                mu, std = policy(state_tensor)
                # print(prob.shape)
                n = Normal(mu, std)
                log_prob = n.log_prob(action_tensor)
                old_log_prob = old_n.log_prob(action_tensor)
                ratio = torch.exp(log_prob - old_log_prob)
                # print(ratio.shape, log_prob.shape)
                L1 = ratio * advantage
                L2 = torch.clamp(ratio, 0.8, 1.2) * advantage
                # print(log_prob.shape)
                loss = torch.min(L1, L2)
                loss = - loss.mean()
                writer.add_scalar('action loss', loss.item(), steps)
                # print(loss.shape)
                optim.zero_grad()
                loss.backward()
                optim.step()

                value_loss = F.mse_loss(rewards_tensor, value(state_tensor))
                value_optim.zero_grad()
                value_loss.backward()
                value_optim.step()
                writer.add_scalar('value loss', value_loss.item(), steps)
            rewards = []
            states = []
            actions = []

    writer.add_scalar('episode reward', episode_reward, epoch)
    if epoch % 10 == 0:
        print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
        torch.save(policy.state_dict(), 'ppo-policy.para')


