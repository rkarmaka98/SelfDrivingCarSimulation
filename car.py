# car.py
import pygame

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 50
        self.height = 100
        self.velocity = 10  # Movement speed

    def draw(self, win):
        pygame.draw.rect(win, (0, 255, 0), (self.x, self.y, self.width, self.height))
