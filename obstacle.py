import pygame
import random
from config import WIDTH

class Obstacle:
    def __init__(self):
        self.x = random.randint(0, WIDTH - 50)
        self.y = -100  # Start off-screen
        self.width = 50
        self.height = 50
        self.speed = 2  # Movement speed

    def move(self):
        self.y += self.speed

    def draw(self, win):
        pygame.draw.rect(win, (255, 0, 0), (self.x, self.y, self.width, self.height))
import pygame  # Import the 'pygame' library to handle graphics and game functionalities.
import random  # Import the 'random' module to generate random numbers.
from config import WIDTH  # Import the 'WIDTH' constant from a 'config' module to use for defining boundaries.

# Define the 'Obstacle' class to represent obstacles that move down the screen.
class Obstacle:
    # The constructor initializes the attributes for each obstacle.
    def __init__(self):
        # Set the initial x-coordinate of the obstacle randomly within the screen width, minus the obstacle width to avoid going out of bounds.
        self.x = random.randint(0, WIDTH - 50)
        
        # Set the initial y-coordinate to start off-screen at -100 pixels, allowing it to enter from the top.
        self.y = -100
        
        # Set the dimensions of the obstacle. Both width and height are 50 pixels, making it a square shape.
        self.width = 50
        self.height = 50
        
        # Set the speed at which the obstacle moves down the screen. A value of 2 means it moves 2 pixels each frame.
        self.speed = 2

    # Define a method to move the obstacle downwards.
    def move(self):
        # Increase the y-coordinate by the speed to make the obstacle move downward.
        self.y += self.speed

    # Define a method to draw the obstacle on the game window.
    # The 'win' parameter is the surface where the obstacle will be drawn.
    def draw(self, win):
        # Use 'pygame.draw.rect' to draw the obstacle as a red rectangle.
        # Arguments:
        # 1. 'win': The surface on which to draw the obstacle.
        # 2. (255, 0, 0): The RGB color of the obstacle (red).
        # 3. (self.x, self.y, self.width, self.height): The position and dimensions of the rectangle.
        pygame.draw.rect(win, (255, 0, 0), (self.x, self.y, self.width, self.height))
