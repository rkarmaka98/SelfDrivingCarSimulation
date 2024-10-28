# train.py
import pygame
import sys
from game_env import GameEnv
from dqn_agent import DQNAgent
from config import CLOCK
import numpy as np
import torch

def train_agent(episodes=1000):
    env = GameEnv()
    state_size = len(env.get_state())
    action_size = 3  # Left, Stay, Right
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        for time_t in range(500):  # Max steps per episode
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Uncomment to render during training (will slow down training)
            # env.render()

            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Limit the frame rate
            CLOCK.tick(60)

        # Save the model every 50 episodes
        if e % 50 == 0:
            agent.save('dqn_self_driving_car.pth')

    # Save the trained model
    agent.save('dqn_self_driving_car.pth')
    pygame.quit()
