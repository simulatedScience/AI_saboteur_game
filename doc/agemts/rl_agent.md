# RL Agent Training and Design for Saboteur

## Overview

This document describes the design and training approach for reinforcement learning (RL) agents in the Saboteur game. Our aim is to develop agents that learn to play the game in a multi-agent setting and can be later used in the GUI (by selecting an "rl_agent" player type or loading a saved model).

## State Representation

We encode the game state as a dense vector that consists of two parts:
1. **Board Encoding (225 dimensions):**  
   - We fix a grid of size 15×15 (covering coordinates from -7 to +7 in both directions, with the start tile at (0,0)).
   - Each cell is encoded as:
     - `0`: Empty.
     - `1`: Start card.
     - `2`: Path card.
     - `3`: Uncovered goal card.
     - `4`: Hidden goal card.
   - The 15×15 grid is then flattened into a 225-dimensional vector.

2. **Hand Encoding (H dimensions):**  
   - For the current player's hand (of fixed size H, e.g. 6), each card is encoded as:
     - `1`: Start card.
     - `2`: Path card.
     - `3`: Uncovered goal card.
     - `4`: Hidden goal card.
   - This yields an H-dimensional vector.

The final state vector is the concatenation of the board encoding and hand encoding (total dimension = 225 + H).

## Action Space

An action in Saboteur consists of:
- **Card index:** Which card in the hand to play (0 to H-1).
- **Board position:** Where to place it. We discretize the board grid (15×15 cells). The grid index (0 to 14 for both x and y) is mapped to board coordinates by subtracting 7.
- **Orientation:** Either 0° (encoded as 0) or 180° (encoded as 1).

Thus, the total number of discrete actions is:  
`H * 15 * 15 * 2`

An action is represented as a single integer in this range. The agent uses an inverse mapping to convert this integer to a triplet (card_index, board_position, orientation).

## Network Architecture

We implement a Deep Q-Network (DQN) with fully connected layers. The architecture is configurable via a tuple parameter:
- **Input Layer:** Size equal to the state dimension (e.g., 231 if H=6).
- **Hidden Layers:** A configurable tuple (e.g., (128, 128)). Each hidden layer uses ReLU activation.
- **Output Layer:** Size equal to the discrete action space (e.g., 6 * 15 * 15 * 2 = 2700).

This DQN outputs Q-values for each discrete action.

## Training Procedure

1. **Experience Collection:**  
   The agent interacts with the Saboteur environment (SaboteurEnv) and collects transitions in the form `(state, action, reward, next_state, done)`.  
   The state is computed as described above; the action is stored as its discrete integer index.

2. **Replay Buffer:**  
   Transitions are stored in a replay buffer. A batch of transitions is sampled for training.

3. **Optimization:**  
   We use a mean squared error loss between the predicted Q-values and the target Q-values computed using the Bellman equation.  
   A target network is used and updated periodically.

4. **Epsilon-Greedy:**  
   The agent uses an epsilon-greedy policy for exploration during training.

5. **Hyperparameters:**  
   Hyperparameters such as learning rate, discount factor, batch size, and network architecture are configurable via the `AI_CONFIG` dictionary.

6. **Multi-Agent Considerations:**  
   Initially, training is done in a self-play (single-agent) setup. Later, the framework can be extended to multi-agent training where different networks interact.

## How to Train

- Run the training script (`train_rl.py`).  
- The training script resets the environment at the beginning of each episode, collects transitions, and updates the network parameters using batches sampled from the replay buffer.  
- Periodically, the target network is updated.
- When training is complete, the trained model is saved (e.g., `trained_dqn.pt`).

## Using the Trained Agent in the GUI

A new player type (e.g., "rl_agent") can be added in the configuration. The GUI can load a stored neural network (from a file such as `trained_dqn.pt`) and create an RL agent instance that uses the trained DQN to select moves automatically.

## Next Steps

- Refine the state representation to include more detailed information (such as edge connectivity, card rotation, etc.).
- Improve the action mapping (and record the discrete action index during training).
- Extend the training to multi-agent setups.
- Experiment with convolutional architectures if necessary.

Happy training!
