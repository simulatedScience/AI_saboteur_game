# Saboteur RL Agent Training Guide

## Overview

This guide describes how to train an RL agent for the Saboteur game using a hybrid action space.
The agent must choose a card from its hand, output continuous (x, y) coordinates for placement (which
are snapped to the nearest valid position), and select an orientation (0° or 180°).

## State Representation

The state vector is composed of:
- **Board State (2200 features):**  
  Up to 100 board cards are encoded. Each card is represented by 22 features:
  - **Position (2 features):** Continuous (x, y) coordinates.
  - **Edge Types (12 features):** For each of the four edges (top, right, bottom, left), one-hot encoding for the edge type (wall, path, dead-end). (We treat path and dead-end similarly for placement, but the connection bits capture finer differences.)
  - **Connection Bits (6 features):** Binary indicators for the presence of each of 6 possible connection pairs.
  - **Special Flags (2 features):** A hidden goal flag (1 if the card is a hidden goal) and a start flag (1 if the card is the start).
- **Hand State (132 features):**  
  The current player's hand of 6 cards is encoded in the same way (6 × 22).

**Total state vector length:** 2200 + 132 = **2332**.

## Action Space

A valid move in Saboteur consists of:
1. **Card Selection:**  
   Choose one card from the hand (Discrete: 0–5).
2. **Placement Coordinates:**  
   Output continuous (x, y) coordinates (each in [-10, 10]). These coordinates are then “snapped” by the environment to the closest valid placement.
3. **Orientation:**  
   Choose the card’s orientation (Discrete: 0 for 0°, 1 for 180°).

The action is represented as a dictionary with keys:
- `"card"`: Discrete
- `"coord"`: Continuous Box(2,)
- `"orient"`: Discrete

## Network Architecture

Our policy network takes the 2332-dimensional state vector as input and uses configurable hidden layers (e.g., `(256, 256)`). It outputs three heads:
- **Card Selection Head:** Outputs logits for 6 choices.
- **Placement Coordinate Head:** Outputs a 2-dimensional continuous vector.
- **Orientation Head:** Outputs logits for 2 choices.

## Training with SB3

We use Stable-Baselines3’s MaskablePPO to train the agent. The training pipeline includes:
- A custom environment wrapper (**SaboteurHybridWrapper**) that provides the state vector and a dictionary action space.
- An action masker wrapper (using a dummy mask function, to be refined later) that can restrict actions to only valid moves.
- TensorBoard logging is enabled. Each training run creates a new folder (named with the current datetime) in which:
  - The model and checkpoints are saved.
  - A `training_info.md` file logs hyperparameters and configuration settings.

## How to Start Training

1. **Install Dependencies:**  
   Ensure you have Python 3.11 and install the following packages:
   ```bash
   pip install stable-baselines3 sb3-contrib torch gymnasium
   ```

## Run the Training Script:
   Execute:
   ```bash
   python train_sb3.py
   ```
A new folder will be created under training_runs/ (e.g., `training_runs/20230412_153000`) where the model, checkpoints, and `training_info.md` are saved.

## Monitor Training:
Launch TensorBoard by running:
```bash
tensorboard --logdir training_runs
```
Open the provided URL in your browser to track training progress.

## Using the Trained Agent:
After training, the model is saved (e.g., as `trained_model.zip`). This model can be loaded in the GUI by specifying an agent type such as `"rl_agent"`.

## Hyperparameters and Configuration
Key parameters are defined in the configuration files:

- **`CONFIG:`** Contains game parameters (e.g., hand_size).
- **`AI_CONFIG:`** Contains RL hyperparameters:
    - `lr`: Learning rate (default: 0.001)
    - `dqn_hidden_layers`: Hidden layer sizes (default: (256, 256))
    - `epsilon`: Exploration rate (default: 0.1)
    - `timesteps`: Total training timesteps (default: 100000)

All these settings are recorded in the `training_info.md` file of each run.
