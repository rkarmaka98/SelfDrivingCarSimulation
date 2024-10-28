# obstacle.py
import pygame
import random
from config import WIDTH

class Obstacle:
    def __init__(self):
        self.x = random.randint(0, WIDTH - 50)
        self.y = -100  # Start off-screen
        self.width = 50
        self.height = 100
        self.speed = 7  # Movement speed

    def move(self):
        self.y += self.speed

    def draw(self, win):
        pygame.draw.rect(win, (255, 0, 0), (self.x, self.y, self.width, self.height))
