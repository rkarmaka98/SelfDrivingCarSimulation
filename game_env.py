import pygame  # Import 'pygame' for creating the game environment and handling graphics.
import random  # Import 'random' for generating random numbers, useful for spawning obstacles at random positions.
import numpy as np  # Import 'numpy' for handling numeric data and creating arrays, used in state representation.
from car import Car  # Import 'Car' class from car module.
from obstacle import Obstacle  # Import 'Obstacle' class from obstacle module.
from config import WIDTH, HEIGHT, WINDOW  # Import game window properties from 'config' module.
import sys  # Import 'sys' for system-level operations.

# Class to represent the game environment.
class GameEnv:
    def __init__(self):
        # Initialize the car at the center bottom of the screen.
        self.car = Car(WIDTH // 2, HEIGHT - 120)
        
        # List to hold all current obstacles.
        self.obstacles = []
        
        # Initialize score.
        self.score = 0
        
        # Boolean to track if the game is over.
        self.done = False
        
        # Initialize font for displaying text on the screen.
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20)

    def step(self, action):
        # Save the previous x-coordinate of the car to calculate movement-based rewards.
        prev_x = self.car.x

        # Interpret action: 0 = Move left, 1 = Stay, 2 = Move right.
        if action == 0:
            # Move left, ensuring it doesn't go out of bounds.
            self.car.x = max(self.car.x - self.car.velocity, 0)
        elif action == 2:
            # Move right, ensuring it doesn't exceed window width.
            self.car.x = min(self.car.x + self.car.velocity, WIDTH - self.car.width)

        # Spawn new obstacles randomly.
        if random.randint(1, 100) == 1:  # Reduced frequency to make testing easier.
            self.obstacles.append(Obstacle())

        # Move existing obstacles.
        for obstacle in self.obstacles[:]:
            obstacle.move()
            if obstacle.y > HEIGHT:  # Remove obstacle once it goes off-screen.
                self.obstacles.remove(obstacle)
                self.score += 1  # Increment score for successfully avoiding an obstacle.

        # Check for collision between car and obstacles.
        collision = False
        for obstacle in self.obstacles:
            if (self.car.x < obstacle.x + obstacle.width and
                self.car.x + self.car.width > obstacle.x and
                self.car.y < obstacle.y + obstacle.height and
                self.car.y + self.car.height > obstacle.y):
                collision = True
                self.done = True  # Mark the game as over if a collision occurs.
                break

        # Calculate the reward for the step taken.
        reward = self.calculate_reward(action, prev_x, collision)

        # Return the new state, the reward, and the 'done' flag.
        return self.get_state(), reward, self.done

    def calculate_reward(self, action, prev_x, collision):
        reward = 0

        # Negative reward for collision.
        if collision:
            reward -= 100
        else:
            # Positive reward for survival.
            reward += 1

            # Reward based on moving away from the closest obstacle.
            closest_obstacle = self.get_closest_obstacle()
            if closest_obstacle:
                obstacle_center = closest_obstacle.x + closest_obstacle.width / 2
                car_center = self.car.x + self.car.width / 2

                prev_distance = abs(prev_x + self.car.width / 2 - obstacle_center)
                current_distance = abs(car_center - obstacle_center)

                # If the car moves away from the closest obstacle, give an additional reward.
                if current_distance > prev_distance:
                    reward += 2

            # Penalty for staying idle.
            if action == 1:
                reward -= 0.5

        return reward

    def get_closest_obstacle(self):
        # Find the obstacle that is closest to the car in the vertical direction.
        if self.obstacles:
            car_y = self.car.y
            closest_obstacle = min(
                self.obstacles, key=lambda obs: abs(obs.y + obs.height - car_y)
            )
            return closest_obstacle
        else:
            return None    

    def get_state(self):
        # Represent the state of the environment, including the car position and nearby obstacles.
        state = [
            self.car.x / WIDTH,  # Normalized car x-coordinate.
            self.car.y / HEIGHT  # Normalized car y-coordinate.
        ]

        # Include the positions of up to 5 closest obstacles.
        obstacles = sorted(self.obstacles, key=lambda obs: abs(obs.y - self.car.y))
        for i in range(10):
            if i < len(obstacles):
                obs = obstacles[i]
                state.extend([
                    obs.x / WIDTH,  # Normalized x-coordinate of the obstacle.
                    obs.y / HEIGHT,  # Normalized y-coordinate of the obstacle.
                    (obs.x - self.car.x) / WIDTH,  # Relative x distance.
                    (obs.y - self.car.y) / HEIGHT,  # Relative y distance.
                ])
            else:
                state.extend([0, 0, 0, 0])  # Default values if no obstacle.

        return np.array(state, dtype=np.float32)

    def reset(self):
        # Reset the environment to its initial state.
        self.__init__()
        return self.get_state()

    def render(self, action=None, q_values=None):
        # Fill the window with a black background.
        WINDOW.fill((0, 0, 0))
        
        # Draw the car on the window.
        self.car.draw(WINDOW)
        
        # Draw all obstacles.
        for obstacle in self.obstacles:
            obstacle.draw(WINDOW)

        # Draw sensor lines (for visualization purposes).
        self.draw_sensors()

        # Display the current action taken.
        if action is not None:
            action_text = f"Action: {['Left', 'Stay', 'Right'][action]}"
            text_surface = self.font.render(action_text, True, (255, 255, 255))
            WINDOW.blit(text_surface, (10, 10))

        # Display the Q-values if provided.
        if q_values is not None:
            q_text = f"Q-values: Left={q_values[0]:.2f}, Stay={q_values[1]:.2f}, Right={q_values[2]:.2f}"
            q_surface = self.font.render(q_text, True, (255, 255, 255))
            WINDOW.blit(q_surface, (10, 40))

        # Update the display to reflect all changes.
        pygame.display.update()

    def draw_sensors(self):
        # Draw three sensor lines ahead of the car.
        sensor_length = 200
        sensor_color = (0, 255, 255)  # Cyan color for sensors.
        car_center_x = self.car.x + self.car.width / 2
        car_front_y = self.car.y

        # Draw a straight-ahead sensor line.
        pygame.draw.line(WINDOW, sensor_color,
                         (car_center_x, car_front_y),
                         (car_center_x, max(car_front_y - sensor_length, 0)), 2)

        # Draw a left-angled sensor line.
        pygame.draw.line(WINDOW, sensor_color,
                         (car_center_x, car_front_y),
                         (max(car_center_x - sensor_length / 2, 0), max(car_front_y - sensor_length, 0)), 2)

        # Draw a right-angled sensor line.
        pygame.draw.line(WINDOW, sensor_color,
                         (car_center_x, car_front_y),
                         (min(car_center_x + sensor_length / 2, WIDTH), max(car_front_y - sensor_length, 0)), 2)
