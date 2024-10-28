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
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20)

    def step(self, action):
        # Action: 0 = Left, 1 = Stay, 2 = Right
        if action == 0:
            self.car.x = max(self.car.x - self.car.velocity, 0)
        elif action == 2:
            self.car.x = min(self.car.x + self.car.velocity, WIDTH - self.car.width)

        # Spawn Obstacles
        if random.randint(1, 60) == 1:  # Adjusted frequency for testing
            self.obstacles.append(Obstacle())

        # Move Obstacles
        for obstacle in self.obstacles[:]:
            obstacle.move()
            if obstacle.y > HEIGHT:
                self.obstacles.remove(obstacle)
                self.score += 10  # Reward for passing an obstacle

            # Collision Detection
            collision_margin = 5  # Adjust as needed
            if (self.car.x + collision_margin < obstacle.x + obstacle.width - collision_margin and
                self.car.x + self.car.width - collision_margin > obstacle.x + collision_margin and
                self.car.y + collision_margin < obstacle.y + obstacle.height - collision_margin and
                self.car.y + self.car.height - collision_margin > obstacle.y + collision_margin):
                self.done = True
                reward = -100
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

    def render(self, action=None, q_values=None):
        WINDOW.fill((0, 0, 0))
        self.car.draw(WINDOW)
        for obstacle in self.obstacles:
            obstacle.draw(WINDOW)

        # Draw sensor lines
        self.draw_sensors()

        # Display action
        if action is not None:
            action_text = f"Action: {['Left', 'Stay', 'Right'][action]}"
            text_surface = self.font.render(action_text, True, (255, 255, 255))
            WINDOW.blit(text_surface, (10, 10))

        # Display Q-values
        if q_values is not None:
            q_text = f"Q-values: Left={q_values[0]:.2f}, Stay={q_values[1]:.2f}, Right={q_values[2]:.2f}"
            q_surface = self.font.render(q_text, True, (255, 255, 255))
            WINDOW.blit(q_surface, (10, 40))

        pygame.display.update()

        def draw_sensors(self):
            # Example: Draw lines ahead of the car
            sensor_length = 200
            sensor_color = (0, 255, 255)  # Cyan color
            car_center_x = self.car.x + self.car.width / 2
            car_front_y = self.car.y

            # Straight ahead
            pygame.draw.line(WINDOW, sensor_color,
                            (car_center_x, car_front_y),
                            (car_center_x, max(car_front_y - sensor_length, 0)), 2)

            # Left sensor
            pygame.draw.line(WINDOW, sensor_color,
                            (car_center_x, car_front_y),
                            (max(car_center_x - sensor_length / 2, 0), max(car_front_y - sensor_length, 0)), 2)

            # Right sensor
            pygame.draw.line(WINDOW, sensor_color,
                            (car_center_x, car_front_y),
                            (min(car_center_x + sensor_length / 2, WIDTH), max(car_front_y - sensor_length, 0)), 2)