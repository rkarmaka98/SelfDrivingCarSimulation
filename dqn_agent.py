import random  # Import 'random' for generating random samples.
import numpy as np  # Import 'numpy' for working with arrays and numerical data.
from collections import deque  # Import 'deque' from 'collections' to create a double-ended queue.
import torch  # Import 'torch' for PyTorch functionalities.
import torch.nn as nn  # Import the 'nn' module for building neural network layers.
import torch.optim as optim  # Import the 'optim' module for optimization algorithms.
from model import DQNModel  # Import 'DQNModel' which defines the architecture of the neural network.
from torch.cuda.amp import autocast, GradScaler  # Import for mixed precision training to improve performance on GPUs.

# Define the device to use GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class definition for the Deep Q-Network (DQN) agent.
class DQNAgent:
    def __init__(self, state_size, action_size):
        # Initialize the size of the state space and action space.
        self.state_size = state_size
        self.action_size = action_size
        
        # Create a replay memory buffer using deque to store experiences, with a maximum length of 2000.
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters for the DQN agent.
        self.gamma = 0.99  # Discount rate for future rewards.
        self.epsilon = 1.0  # Initial exploration rate.
        self.epsilon_min = 0.01  # Minimum value for epsilon to ensure some exploration.
        self.epsilon_decay = 0.995  # Rate at which epsilon decays over time to reduce exploration.
        
        # Initialize the Q-network using the DQNModel architecture.
        self.model = DQNModel(state_size, action_size).to(device)
        
        # Define the optimizer for training the model.
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        
        # Define the loss function to be Mean Squared Error (MSE).
        self.criterion = nn.MSELoss()
        
        # Initialize a gradient scaler for mixed precision training to speed up training on the GPU.
        self.scaler = GradScaler()

    # Store experiences (state, action, reward, next_state, done) in the replay memory.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose an action based on an epsilon-greedy policy.
    def act(self, state):
        # Epsilon-greedy: with probability epsilon, select a random action (exploration).
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            # Convert the state to a torch tensor and send it to the device (CPU or GPU).
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # Add batch dimension.

            # Set the model to evaluation mode to disable dropout, etc.
            self.model.eval()
            
            with torch.no_grad():
                # Get the action values predicted by the model for the given state.
                act_values = self.model(state)
            
            # Get the index of the action with the highest Q-value.
            action = torch.argmax(act_values).item()
            
            # Set the model back to training mode.
            self.model.train()
        
        return action
        
    # Train the model by replaying a batch of experiences.
    def replay(self, batch_size):
        # Sample a minibatch of experiences from the replay memory.
        minibatch = random.sample(self.memory, batch_size)
        
        # Extract states, actions, rewards, next_states, and done flags from the minibatch.
        states = np.vstack([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.vstack([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        # Convert the arrays to PyTorch tensors and move them to the device.
        states = torch.from_numpy(states).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        dones = torch.from_numpy(dones.astype(int)).float().to(device)

        # Enable mixed precision training with autocast.
        with autocast():
            # Calculate Q-values for the current states for the selected actions.
            q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Calculate the target Q-values for the next states using the Bellman equation.
            with torch.no_grad():
                next_q_values = self.model(next_states).max(1)[0]

            # Calculate target values using rewards and discounted next Q-values.
            targets = rewards + (self.gamma * next_q_values * (1 - dones))

            # Calculate the loss between the predicted Q-values and the target Q-values.
            loss = self.criterion(q_values, targets)

        # Optimize the model to minimize the loss.
        self.optimizer.zero_grad()  # Zero the gradients before backward pass.
        self.scaler.scale(loss).backward()  # Use the scaler to backward propagate the loss.
        self.scaler.step(self.optimizer)  # Update the model parameters.
        self.scaler.update()  # Update the scaler for mixed precision.

        # Reduce the exploration rate gradually, ensuring it does not go below the minimum value.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Load the model's state dictionary from a saved file.
    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    # Save the model's state dictionary to a file.
    def save(self, name):
        torch.save(self.model.state_dict(), name)
