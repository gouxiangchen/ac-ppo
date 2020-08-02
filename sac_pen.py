import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import numpy as np
from collections import deque
import random
from itertools import count
import gym


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


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


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256 + 1, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = self.relu(self.fc1(x))
        x = torch.cat([x, a], dim=1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, 1)
        self.fc_log_std = nn.Linear(128, 1)

        self.action_scale = 2.

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)

        std = torch.exp(log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = self.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1-y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = self.tanh(mean) * self.action_scale

        return action, log_prob, mean

    def choose_action(self, x):
        x = torch.FloatTensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, mean = self.forward(x)

        return action.item(), log_prob, mean.item()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


if __name__ == "__main__":
    
    critic_1 = Critic().to(device)
    critic_2 = Critic().to(device)

    critic_1_target = Critic().to(device)
    critic_2_target = Critic().to(device)

    critic_1_optim = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2_optim = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    actor = Actor().to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)

    gamma = 0.99
    alpha = 0.2
    tau = 0.005

    target_entropy = -1.
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=3e-4)

    memory_replay = Memory(50000)
    begin_train = False
    batch_size = 32

    learn_steps = 0

    writer = SummaryWriter('sac_original')

    env = gym.make('Pendulum-v0')

    env_steps = 0

    for epoch in count():
        episode_reward = 0.
        state = env.reset()
        for time_steps in range(200):
            env_steps += 1
            action, log_prob, _ = actor.choose_action(state)
            writer.add_scalar('log prob', torch.exp(log_prob).item(), env_steps)
            next_state, reward, done, _ = env.step([action])
            episode_reward += reward
            reward = (reward + 8.1) / 8.1
            memory_replay.add((state, next_state, action, reward))
            state = next_state
            if memory_replay.size() > 1280:
                learn_steps += 1
                if not begin_train:
                    print('train begin!')
                    begin_train = True
                
                alpha = log_alpha.exp().detach()

                experiences = memory_replay.sample(batch_size, False)
                batch_state, batch_next_state, batch_action, batch_reward = zip(*experiences)

                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)

                with torch.no_grad():
                    next_state_action, next_state_log_pi, _ = actor(batch_next_state)
                    qf1_next_target = critic_1_target(batch_next_state, next_state_action)
                    qf2_next_target = critic_2_target(batch_next_state, next_state_action)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = batch_reward + gamma * (min_qf_next_target)

                critic_1_loss = F.mse_loss(critic_1(batch_state, batch_action), next_q_value)
                critic_1_optim.zero_grad()
                critic_1_loss.backward()
                critic_1_optim.step()

                critic_2_loss = F.mse_loss(critic_2(batch_state, batch_action), next_q_value)
                critic_2_optim.zero_grad()
                critic_2_loss.backward()
                critic_2_optim.step()

                pi, log_pi, _ = actor(batch_state)
                qf1_pi = critic_1(batch_state, pi)
                qf2_pi = critic_2(batch_state, pi)

                actor_loss = ((alpha * log_pi) - torch.min(qf1_pi, qf2_pi)).mean()
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()
                writer.add_scalar('alpha loss', alpha_loss.item(), learn_steps)
                writer.add_scalar('alpha', log_alpha.exp().item(), learn_steps)

                writer.add_scalar('critic 1 loss', critic_1_loss.item(), learn_steps)
                writer.add_scalar('critic 2 loss', critic_2_loss.item(), learn_steps)
                writer.add_scalar('actor loss', actor_loss.item(), learn_steps)

                soft_update(critic_1_target, critic_1, tau)
                soft_update(critic_2_target, critic_2, tau)

        writer.add_scalar('episode reward', episode_reward, epoch)
        if epoch % 1 == 0:
            print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))

        if epoch % 100 == 0:
            torch.save(actor.state_dict(), 'sac/sac-actor-soft.para')
            print('model saved!')


