import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from config import device


class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class PrioritizedReplayBuffer:
    def __init__(self, state_dim, alpha=0.6, beta=0.4, beta_increment=0.001, max_capacity=100000):
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_capacity = max_capacity
        self.size = 0
        self.next_ind = 0

        self.tree_size = 1
        while self.tree_size < self.max_capacity:
            self.tree_size <<= 1

        self.sum_tree = np.zeros(2 * self.tree_size)
        self.min_tree = np.full(2 * self.tree_size, float('inf'))
        self.max_priority = 1.0

        self.data = {
            'state': np.zeros((max_capacity, state_dim), dtype=np.float32),
            'action': np.zeros(max_capacity, dtype=np.int64),
            'reward': np.zeros(max_capacity, dtype=np.float32),
            'next_state': np.zeros((max_capacity, state_dim), dtype=np.float32),
            'done': np.zeros(max_capacity, dtype=np.bool_)
        }

    def add(self, state, action, reward, next_state, done):
        ind = self.next_ind
        self.data['state'][ind] = state
        self.data['action'][ind] = action
        self.data['reward'][ind] = reward
        self.data['next_state'][ind] = next_state
        self.data['done'][ind] = done

        self.next_ind = (ind + 1) % self.max_capacity
        self.size = min(self.size + 1, self.max_capacity)

        priority = self.max_priority ** self.alpha
        self._update_trees(ind, priority)

    def _update_trees(self, ind, priority):
        tree_ind = ind + self.tree_size
        self.sum_tree[tree_ind] = priority
        self.min_tree[tree_ind] = priority

        tree_ind >>= 1
        while tree_ind >= 1:
            left = tree_ind << 1
            right = left + 1
            self.sum_tree[tree_ind] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[tree_ind] = min(self.min_tree[left], self.min_tree[right])
            tree_ind >>= 1

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError("缓冲区数据不足，无法采样")

        samples = {
            'indices': np.zeros(batch_size, dtype=np.int64),
            'weights': np.zeros(batch_size, dtype=np.float32),
            'state': np.zeros((batch_size, self.data['state'].shape[1]), dtype=np.float32),
            'action': np.zeros(batch_size, dtype=np.int64),
            'reward': np.zeros(batch_size, dtype=np.float32),
            'next_state': np.zeros((batch_size, self.data['state'].shape[1]), dtype=np.float32),
            'done': np.zeros(batch_size, dtype=np.bool_)
        }

        total_priority = self.sum_tree[1] + 1e-10
        segment = total_priority / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        min_priority_raw = self.min_tree[1]
        if min_priority_raw == float('inf') or min_priority_raw == 0:
            min_priority_raw = np.min([self.sum_tree[self.tree_size + i] for i in range(self.size)])

        min_priority = (min_priority_raw + 1e-10) / total_priority
        max_weight = (self.size * min_priority) ** (-self.beta) if min_priority != 0 else 1.0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            p = random.uniform(a, b)

            ind = self._find_index(p)
            ind = max(0, min(ind, self.size - 1))
            samples['indices'][i] = ind

            priority = self.sum_tree[ind + self.tree_size] + 1e-10
            prob = priority / total_priority
            weight = (self.size * prob) ** (-self.beta)
            samples['weights'][i] = weight / max_weight if max_weight != 0 else weight

            samples['state'][i] = self.data['state'][ind]
            samples['action'][i] = self.data['action'][ind]
            samples['reward'][i] = self.data['reward'][ind]
            samples['next_state'][i] = self.data['next_state'][ind]
            samples['done'][i] = self.data['done'][ind]

        return samples

    def _find_index(self, p):
        ind = 1
        while ind < self.tree_size:
            left = ind << 1
            right = left + 1
            if p <= self.sum_tree[left]:
                ind = left
            else:
                p -= self.sum_tree[left]
                ind = right
        return ind - self.tree_size

    def update_priorities(self, indices, priorities):
        for ind, priority in zip(indices, priorities):
            adjusted_priority = priority + 1e-6
            if adjusted_priority > self.max_priority:
                self.max_priority = adjusted_priority
            self._update_trees(ind, adjusted_priority ** self.alpha)

    def __len__(self):
        return self.size


class DuelingDDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.batch_size = 128
        self.tau = 0.005
        self.lr = 1e-3
        self.update_interval = 10
        self.step_counter = 0

        self.policy_net = DuelingNetwork(state_dim, action_dim).to(device)
        self.target_net = DuelingNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.memory = PrioritizedReplayBuffer(
            state_dim=state_dim,
            alpha=0.6,
            beta=0.4,
            max_capacity=100000
        )

    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_model(self):
        self.step_counter += 1
        if self.step_counter % self.update_interval != 0:
            return

        if len(self.memory) < self.batch_size:
            return

        samples = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(samples['state']).to(device)
        actions = torch.LongTensor(samples['action']).to(device).unsqueeze(1)
        rewards = torch.FloatTensor(samples['reward']).to(device).unsqueeze(1)
        next_states = torch.FloatTensor(samples['next_state']).to(device)
        dones = torch.FloatTensor(samples['done']).to(device).unsqueeze(1)
        weights = torch.FloatTensor(samples['weights']).to(device).unsqueeze(1)
        indices = samples['indices']

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy().flatten()
        loss = (weights * nn.MSELoss(reduction='none')(current_q, target_q)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)
        self.soft_update_target()

    def soft_update_target(self):
        for target_param, policy_param in zip(
                self.target_net.parameters(),
                self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())