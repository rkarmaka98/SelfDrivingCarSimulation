# Self-Driving Car Simulation with Reinforcement Learning

https://github.com/user-attachments/assets/7146335e-90e0-40cf-b844-18543974753b

Welcome to the **Self-Driving Car Simulation** project! This project leverages reinforcement learning to train an agent to autonomously navigate a simulated car through a multi-lane environment while avoiding obstacles. The simulation environment is built using **Pygame** and **OpenAI Gym**, and the agent is implemented using **PyTorch**.


## Overview

The **Self-Driving Car Simulation** project aims to create an environment where an agent learns to drive a car autonomously. The agent receives visual inputs from the environment, processes them through a Deep Q-Network (DQN), and learns optimal actions to maximize its reward, which is primarily based on avoiding collisions and successfully navigating through lanes.

## Features

- **Custom Pygame Environment**: Simulates a 4-lane road with dynamic obstacle generation.
- **Reinforcement Learning Agent**: Implements a DQN agent using PyTorch.
- **Frame Stacking**: Utilizes frame stacking to provide temporal context to the agent.
- **Experience Replay**: Incorporates experience replay to stabilize and improve learning.
- **Target Network**: Employs a target network to enhance training stability.
- **TensorBoard Integration**: Visualizes training metrics for better monitoring.
- **Flexible Configuration**: Easily adjustable parameters for customization.

## Architecture

The project is structured into several key components:

- **Environment (`game_env.py`)**: Defines the simulation environment using Pygame and adheres to the OpenAI Gym interface.
- **Agent (`dqn_agent.py`)**: Implements the DQN agent responsible for decision-making and learning.
- **Model (`model.py`)**: Defines the neural network architecture used by the agent.
- **Training Script (`train.py`)**: Facilitates the training process, including environment interaction and agent updates.
- **Simulation Script (`run_simulation.py`)**: Runs the trained agent in the environment for evaluation and visualization.
- **Configuration (`config.py`)**: Contains configurable parameters for the environment and agent.
- **Supporting Classes (`car.py`, `obstacle.py`)**: Define the car and obstacle entities within the simulation.

## Installation

### Prerequisites

- **Python 3.7+**
- **pip** (Python package manager)

### Clone the Repository

```
git clone https://github.com/yourusername/SelfDrivingCarSimulation.git
cd SelfDrivingCarSimulation
```
## Usage
### Training the Agent
Training the agent involves running the train.py script, which will initiate the training loop where the agent interacts with the environment, learns from experiences, and updates its neural network accordingly.
```
python train.py
```
### Optional Arguments:

`--episodes`: Number of training episodes (default: 1000)
`--batch_size`: Size of experience replay batch (default: 64)
`--render`: Enable environment rendering during training (default: False)
### Example:

```
python train.py --episodes 5000 --batch_size 128 --render
```
Note: Rendering during training can significantly slow down the process. It's recommended to disable rendering unless debugging.

### Running the Simulation
After training, you can run the simulation to observe the agent's performance in navigating the environment.

```
python run_simulation.py
```
The simulation will load the trained model and display the agent's actions in real-time. The window will show the car's movements and obstacles. Press the close button on the window or let the simulation run until a collision occurs.

## Configuration
All configurable parameters are located in the config.py file. You can adjust settings such as screen dimensions, number of lanes, obstacle spawn probability, and cooldown periods.
```
# Screen dimensions (scaled up before resizing)
WIDTH = 840  # 84 * 10
HEIGHT = 588  # 84 * 7

# Number of lanes
NUM_LANES = 4

# Lane width
LANE_WIDTH = WIDTH // NUM_LANES

# Define lane centers
LANE_CENTER_X = [LANE_WIDTH * i + LANE_WIDTH // 2 for i in range(NUM_LANES)]

# Obstacle spawn configurations
OBSTACLE_SPAWN_PROBABILITY = 1 / 100  # Adjusted spawn probability
OBSTACLE_SPAWN_COOLDOWN = {
    lane: 60 * 2  # Cooldown in frames (e.g., 2 seconds at 60 FPS)
    for lane in range(NUM_LANES)
}

Key Parameters:

WIDTH & HEIGHT: Define the size of the simulation window.
NUM_LANES: Number of lanes on the road.
LANE_WIDTH: Calculated based on the number of lanes.
OBSTACLE_SPAWN_PROBABILITY: Probability of spawning an obstacle in a lane.
OBSTACLE_SPAWN_COOLDOWN: Frames to wait before spawning another obstacle in a lane.
```
## Troubleshooting

### Issue: Training is significantly slower when rendering is enabled.
Solution: Disable rendering during training by ensuring the render flag is set to False when calling train_agent().
### Debugging Tips
Print Statements: Use print statements or logging to verify the shapes of frames and tensors at various stages.

```
print(f"Stacked Frames Shape: {stacked_frames.shape}")  # Should be (4, 84, 84)
print(f"Replay Batch States Shape: {states.shape}")    # Should be (batch_size, 4, 84, 84)
Assertions: Utilize assertions to enforce expected shapes and catch discrepancies early.
```

```
assert state.shape == self.state_size, f"Expected state shape {self.state_size}, got {state.shape}"
```
## Contributing
Contributions are welcome! Whether you're reporting bugs, suggesting enhancements, or submitting pull requests, your participation is invaluable.

Fork the Repository

- Create a Feature Branch
`git checkout -b feature/YourFeature`

- Commit Your Changes
`git commit -m "Add your feature"`

- Push to the Branch
`git push origin feature/YourFeature`

- Open a Pull Request
Provide a clear description of your changes and the reasoning behind them.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [OpenAI Gym](https://gym.openai.com/) for providing a flexible interface for reinforcement learning environments.
- [Pygame](https://www.pygame.org/wiki/GettingStarted) for enabling easy game development and simulation.
- [PyTorch](https://pytorch.org/) for its robust machine learning library.
