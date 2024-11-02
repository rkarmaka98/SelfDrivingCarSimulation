import sys  # Import the 'sys' module, which provides access to command-line arguments and other system-level functionality.

# The entry point of the script. When this script is executed, the '__name__' variable will be '__main__'.
if __name__ == '__main__':
    
    # Check if the number of command-line arguments is less than 2.
    # The program expects at least one argument in addition to the script name.
    if len(sys.argv) < 2:
        # If the required argument is missing, print a usage message.
        print("Usage: python main.py [train|simulate]")
        # Exit the program with status code 1, indicating an error due to missing arguments.
        sys.exit(1)

    # Check if the first argument is 'train'.
    if sys.argv[1] == 'train':
        # Import the 'train_agent' function from the 'train' module. This import is placed here to avoid unnecessary imports if not training.
        from train import train_agent
        # Call the 'train_agent' function to initiate the training process.
        train_agent()
    # Check if the first argument is 'simulate'.
    elif sys.argv[1] == 'simulate':
        # Import the 'run_simulation' function from the 'run_simulation' module. Again, imported only when needed to save resources.
        from run_simulation import run_simulation
        # Call the 'run_simulation' function to start the simulation process.
        run_simulation()
    # If the argument is neither 'train' nor 'simulate', handle it as an invalid option.
    else:
        # Print an error message indicating an invalid argument and show the valid options.
        print("Invalid option. Use 'train' or 'simulate'.")
        # Exit the program with status code 1 to signal an invalid argument was provided.
        sys.exit(1)
