# game_env.py
import pygame
import random
import numpy as np
from car import Car
from obstacle import Obstacle
from config import WIDTH, HEIGHT, WINDOW
import sys

class GameEnv:
    def __init__(self):
        self.car = Car(WIDTH // 2, HEIGHT - 120)
        self.obstacles = []
        self.score = 0
        self.done = False

    def step(self, action):
        # Action: 0 = Left, 1 = Stay, 2 = Right
        if action == 0 and self.car.x - self.car.velocity >= 0:
            self.car.x -= self.car.velocity
        elif action == 2 and self.car.x + self.car.velocity <= WIDTH - self.car.width:
            self.car.x += self.car.velocity

        # Spawn Obstacles
        if random.randint(1, 60) == 1:  # Adjusted frequency for testing
            self.obstacles.append(Obstacle())

        # Move Obstacles
        for obstacle in self.obstacles[:]:
            obstacle.move()
            if obstacle.y > HEIGHT:
                self.obstacles.remove(obstacle)
                self.score += 1  # Reward for passing an obstacle

            # Collision Detection
            if (self.car.x < obstacle.x + obstacle.width and
                self.car.x + self.car.width > obstacle.x and
                self.car.y < obstacle.y + obstacle.height and
                self.car.y + self.car.height > obstacle.y):
                self.done = True
                reward = -100  # Negative reward on collision
                return self.get_state(), reward, self.done

        reward = 1  # Small reward for each successful move
        return self.get_state(), reward, self.done

    def get_state(self):
        # Simplify the state to the position of the car and obstacles
        state = [
            self.car.x / WIDTH,  # Normalize
            self.car.y / HEIGHT
        ]
        # Include positions of up to 5 nearest obstacles
        obstacles = sorted(self.obstacles, key=lambda obs: obs.y)
        for i in range(5):
            if i < len(obstacles):
                obs = obstacles[i]
                state.extend([
                    obs.x / WIDTH,
                    obs.y / HEIGHT
                ])
            else:
                state.extend([0, 0])
        return np.array(state, dtype=np.float32)

    def reset(self):
        self.__init__()
        return self.get_state()

    def render(self):
        WINDOW.fill((0, 0, 0))
        self.car.draw(WINDOW)
        for obstacle in self.obstacles:
            obstacle.draw(WINDOW)
        pygame.display.update()
