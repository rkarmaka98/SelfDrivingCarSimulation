import pygame  # Importing the 'pygame' library, which is used to create games and graphical applications in Python.

# Defining the 'Car' class to represent a car object.
class Car:
    # Initialization method, called when a new Car instance is created.
    def __init__(self, x, y):
        # Set the initial x and y coordinates of the car. These coordinates determine the position of the car on the screen.
        self.x = x
        self.y = y
        
        # Set the dimensions of the car. The width is 50 pixels and the height is 100 pixels.
        # These attributes define how big the car will be drawn on the screen.
        self.width = 50
        self.height = 100
        
        # Set the velocity of the car, which determines how fast it can move in the game.
        # A value of 10 means the car will move by 10 pixels when it is updated.
        self.velocity = 10

    # Method to draw the car on the game window.
    # 'win' is expected to be a surface object where the car will be drawn.
    def draw(self, win):
        # Use the 'pygame.draw.rect' function to draw a rectangle representing the car.
        # The arguments are:
        # 1. 'win': The surface on which the rectangle (car) will be drawn.
        # 2. (0, 255, 0): The color of the car in RGB format (in this case, green).
        # 3. (self.x, self.y, self.width, self.height): The position and size of the rectangle.
        #    This uses the car's x and y coordinates, and its width and height attributes.
        pygame.draw.rect(win, (0, 255, 0), (self.x, self.y, self.width, self.height))
