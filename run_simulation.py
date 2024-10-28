# run_simulation.py
import pygame
from game_env import GameEnv
from dqn_agent import DQNAgent
from config import CLOCK
import numpy as np
import torch
import sys

def run_simulation():
    env = GameEnv()
    state_size = len(env.get_state())
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    agent.load('dqn_self_driving_car.pth')
    agent.epsilon = 0  # No exploration

    state = env.reset()
    while True:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get action and Q-values
        action, q_values = agent.act(state, return_q_values=True)
        
        # Render the environment with decision info
        env.render(action=action, q_values=q_values)

        next_state, reward, done = env.step(action)
        state = next_state

        if done:
            print("Simulation ended due to collision.")
            pygame.time.wait(2000)
            break

        CLOCK.tick(60)
    pygame.quit()
