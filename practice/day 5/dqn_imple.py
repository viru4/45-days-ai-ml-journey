# dqn_from_scratch.py

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------
# Neural Network (Q-Network)
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------
# Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# -------------------------------
# Hyperparameters
# -------------------------------
state_size = 4
action_size = 2

gamma = 0.99
lr = 0.001
batch_size = 32
buffer_size = 10000

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1

# -------------------------------
# Initialize Networks
# -------------------------------
q_network = QNetwork(state_size, action_size)
target_network = QNetwork(state_size, action_size)

target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=lr)

# -------------------------------
# Replay Buffer
# -------------------------------
memory = ReplayBuffer(buffer_size)

# -------------------------------
# Dummy environment step
# -------------------------------
def step(state, action):
    next_state = np.random.rand(state_size)
    reward = random.random()
    done = random.random() < 0.1
    return next_state, reward, done

# -------------------------------
# Training Loop
# -------------------------------
episodes = 50

for episode in range(episodes):

    state = np.random.rand(state_size)

    for t in range(50):

        # ε-greedy
        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state)
            action = torch.argmax(q_network(state_tensor)).item()

        next_state, reward, done = step(state, action)

        # Store experience
        memory.store((state, action, reward, next_state, done))

        state = next_state

        # Train only if enough samples
        if memory.size() > batch_size:
            batch = memory.sample(batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # Current Q values
            q_values = q_network(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Target Q values
            next_q_values = target_network(next_states).max(1)[0]
            targets = rewards + gamma * next_q_values * (1 - dones)

            # Loss
            loss = nn.MSELoss()(q_values, targets.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network
    if episode % 5 == 0:
        target_network.load_state_dict(q_network.state_dict())

print("Training complete")