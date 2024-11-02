import pygame  # Import 'pygame' for rendering and handling game events.
from game_env import GameEnv  # Import 'GameEnv' for interacting with the game environment.
from dqn_agent import DQNAgent  # Import 'DQNAgent' for the trained deep Q-network agent.
from config import CLOCK, WINDOW  # Import the game clock and window settings from the 'config'.
import numpy as np  # Import 'numpy' for numerical operations.
import torch  # Import PyTorch for handling model inference.
import sys  # Import 'sys' to handle system-level operations like quitting the program.
import time  # Import 'time' for adding delays.

# Define the device for running the model on GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to run the trained agent in a simulation.
def run_simulation():
    # Initialize the game environment.
    env = GameEnv()
    
    # Get the size of the state space from the environment.
    state_size = len(env.get_state())
    
    # Define the number of actions the agent can take: Left, Stay, Right.
    action_size = 3
    
    # Initialize the DQN agent with the state and action sizes.
    agent = DQNAgent(state_size, action_size)
    
    # Load the pre-trained model weights from the saved file.
    agent.load('dqn_self_driving_car.pth')
    
    # Set epsilon to 0 to fully exploit the trained model without exploration.
    agent.epsilon = 0

    # Reset the environment to the initial state.
    state = env.reset()

    # Main loop for running the simulation.
    while True:
        # Render the environment to display the current state.
        env.render()
        
        # Display Q-values in the window to provide insight into the agent's decision process.
        # Convert the state to a tensor and move it to the device (GPU or CPU).
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Compute the Q-values for the current state using the trained model.
        with torch.no_grad():
            q_values = agent.model(state_tensor).cpu().numpy()[0]
        
        # Create a font for displaying the Q-values.
        font = pygame.font.SysFont('Arial', 18)
        
        # Render the Q-values for each action (Left, Stay, Right) as text.
        q_text = f"Q-values: Left={q_values[0]:.2f}, Stay={q_values[1]:.2f}, Right={q_values[2]:.2f}"
        text_surface = font.render(q_text, True, (255, 255, 255))
        WINDOW.blit(text_surface, (10, 10))  # Draw the text at position (10, 10) on the screen.

        # Handle Pygame events (e.g., quitting the game).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()  # Quit Pygame to release resources.
                sys.exit()  # Exit the program.

        # Get the action based on the current policy (pure exploitation since epsilon = 0).
        action = agent.act(state)
        
        # Render the environment, showing the agent's chosen action and Q-values.
        env.render(action=action, q_values=q_values)

        # Take the action in the environment and get the next state, reward, and whether the episode is done.
        next_state, reward, done = env.step(action)
        
        # Update the state to the next state.
        state = next_state

        # If the episode ends (e.g., collision), print a message and wait for 2 seconds.
        if done:
            print("Simulation ended due to collision.")
            pygame.time.wait(2000)
            break

        # Limit the frame rate to 60 frames per second.
        CLOCK.tick(60)

    # Quit Pygame after the simulation is finished.
    pygame.quit()
