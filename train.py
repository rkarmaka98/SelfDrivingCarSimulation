import pygame  # Import 'pygame' for handling events and rendering.
import sys  # Import 'sys' to handle system-level operations like quitting the program.

import torch.backends  # Import PyTorch backend functionalities.
import torch.backends.cuda
import torch.backends.cudnn
from game_env import GameEnv  # Import 'GameEnv' for interacting with the game environment.
from dqn_agent import DQNAgent  # Import 'DQNAgent' for training the deep Q-network agent.
from config import CLOCK  # Import 'CLOCK' for controlling the frame rate.
import numpy as np  # Import 'numpy' for numerical operations.
import torch  # Import PyTorch for tensor operations.

# Enable cuDNN benchmark mode to improve GPU performance if the input sizes are consistent.
torch.backends.cudnn.benchmark = True

# Function to train the DQN agent.
def train_agent(episodes=10000):
    # Initialize the game environment.
    env = GameEnv()
    
    # Get the size of the state space from the environment.
    state_size = len(env.get_state())
    
    # Define the number of possible actions the agent can take: Left, Stay, Right.
    action_size = 3
    
    # Initialize the DQN agent with the state and action sizes.
    agent = DQNAgent(state_size, action_size)
    
    # Set the batch size for experience replay.
    batch_size = 256

    # Loop over the specified number of episodes for training.
    for e in range(episodes):
        # Reset the environment to the initial state at the start of each episode.
        state = env.reset()
        total_reward = 0

        # Each episode runs for a maximum of 1000 time steps.
        for time_t in range(1000):
            # Handle Pygame events to check if the user closes the window.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Quit Pygame and exit the program if the close button is pressed.
                    pygame.quit()
                    sys.exit()

            # The agent selects an action using the current policy (epsilon-greedy).
            action = agent.act(state)
            
            # Perform the action and get the next state, reward, and whether the episode is done.
            next_state, reward, done = env.step(action)
            
            # Accumulate the total reward for the episode.
            total_reward += reward
            
            # Store the experience in replay memory.
            agent.remember(state, action, reward, next_state, done)
            
            # Update the current state to the next state.
            state = next_state

            # Optionally render the environment every 10 episodes (commented out to avoid slowing down training).
            # if e % 10 == 0:
                # env.render()

            # If the episode is done (e.g., the agent hits an obstacle), print episode statistics.
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

            # If there are enough samples in memory, train the agent using experience replay.
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Limit the frame rate to 60 frames per second for consistent gameplay.
            CLOCK.tick(60)

        # Optionally adjust the epsilon decay rate for exploration vs. exploitation.
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Save the model every 50 episodes to ensure progress is saved regularly.
        if e % 50 == 0:
            agent.save('dqn_self_driving_car.pth')

    # Save the trained model after all episodes are completed.
    agent.save('dqn_self_driving_car.pth')
    
    # Quit Pygame to release resources.
    pygame.quit()
