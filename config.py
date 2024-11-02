import pygame  # Import the 'pygame' library for creating graphical applications and handling events.

# Initialize all imported 'pygame' modules. This is required before using most 'pygame' functions.
pygame.init()

# Define the dimensions of the game window.
WIDTH, HEIGHT = 1000, 600  # The game window will be 1000 pixels wide and 600 pixels tall.

# Create the game window using the specified dimensions.
# 'pygame.display.set_mode()' is used to create a window or screen for display.
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

# Set the title of the game window to 'Self-Driving Car Simulation'.
# 'pygame.display.set_caption()' is used to set the window's title bar text.
pygame.display.set_caption('Self-Driving Car Simulation')

# Create a clock object to help control the frame rate of the game.
# 'pygame.time.Clock()' is used to track and control the time between frames, enabling consistent game speed.
CLOCK = pygame.time.Clock()
