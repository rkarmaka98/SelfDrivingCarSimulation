# config.py
import pygame

pygame.init()

WIDTH, HEIGHT = 800, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Self-Driving Car Simulation')
CLOCK = pygame.time.Clock()
