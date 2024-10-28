# dqn_agent.py
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQNModel(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, return_q_values=False):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            q_values = np.zeros(self.action_size)
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values_tensor = self.model(state_tensor)
            q_values = q_values_tensor.cpu().numpy()[0]
            action = np.argmax(q_values)

        if return_q_values:
            return action, q_values
        else:
            return action
        
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.vstack([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        states = torch.from_numpy(states).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        dones = torch.from_numpy(dones.astype(int)).float().to(device)

        # Compute Q values for current states
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Q values for next states
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]

        # Compute target Q values
        targets = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(q_values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reduce exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
