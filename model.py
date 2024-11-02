import torch  # Import PyTorch for building neural networks.
import torch.nn as nn  # Import 'torch.nn' which contains neural network layers and functions.
import torch.nn.functional as F  # Import functional API for activation functions like ReLU.

# Define the neural network model for the Deep Q-Network (DQN).
class DQNModel(nn.Module):
    # The constructor initializes the neural network layers.
    def __init__(self, input_size, output_size):
        # Call the constructor of the superclass (nn.Module).
        super(DQNModel, self).__init__()
        
        # Define the first fully connected layer.
        # Takes 'input_size' features as input and outputs 256 features.
        self.fc1 = nn.Linear(input_size, 256)
        
        # Define the second fully connected layer.
        # Takes the 256 features from the previous layer and outputs another 256 features.
        self.fc2 = nn.Linear(256, 256)
        
        # Define the final fully connected layer.
        # Takes 256 features as input and outputs 'output_size', which corresponds to the number of actions.
        self.fc3 = nn.Linear(256, output_size)

    # Define the forward pass of the neural network.
    # 'x' is the input tensor that will be passed through the layers.
    def forward(self, x):
        # Pass the input 'x' through the first fully connected layer, then apply ReLU activation.
        x = F.relu(self.fc1(x))
        
        # Pass through the second fully connected layer, then apply ReLU activation.
        x = F.relu(self.fc2(x))
        
        # Pass through the final fully connected layer.
        # No activation function here, as we want the raw Q-values for each action.
        x = self.fc3(x)
        
        # Return the output tensor containing Q-values.
        return x
