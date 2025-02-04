# rl_agent.py
"""
This module implements an RL agent for the Saboteur game using a configurable DQN.
It adheres to the BaseAgent interface.
The agent uses a state representation that encodes:
  - A fixed 15x15 board grid (centered at (0,0)) where each cell is:
      0: empty,
      1: start card,
      2: path card,
      3: uncovered goal card,
      4: hidden goal card.
  - The current player's hand encoded as an integer per card:
      1: start, 2: path, 3: uncovered goal, 4: hidden goal.
The discrete action space is defined as:
  - card index (0 to H-1),
  - board cell (x, y) where x and y are in {0,1,...,14} (which map to coordinates by subtracting 7),
  - orientation (0 or 1; 0 means 0° and 1 means 180°).
Total action space size = H * 15 * 15 * 2.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from ..saboteur_env import SaboteurEnv
from ..config import AI_CONFIG, CONFIG

# Use built-in types (Python 3.11 style)

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_layers: tuple) -> None:
        """
        Initialize the DQN model.

        Args:
            state_size (int): Dimension of the input state.
            action_size (int): Number of discrete actions.
            hidden_layers (tuple): A tuple of integers, each specifying the number of units in a hidden layer.
        """
        super().__init__()
        layers = []
        input_size = state_size
        for h in hidden_layers:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h
        layers.append(nn.Linear(input_size, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Q-values for each action.
        """
        return self.model(x)

class RLAgent(BaseAgent):
    def __init__(self, env: SaboteurEnv) -> None:
        """
        Initialize the RLAgent.

        Args:
            env (SaboteurEnv): The game environment.
        """
        super().__init__(env)
        # Define state: we use a 15x15 board grid plus the hand.
        self.grid_size: int = 15
        self.board_state_dim: int = self.grid_size * self.grid_size  # 225 cells
        self.hand_size: int = CONFIG.get("hand_size", 6)
        self.hand_state_dim: int = self.hand_size  # One integer per card
        self.state_size: int = self.board_state_dim + self.hand_state_dim

        # Define action space: hand index * grid_size^2 * 2 orientations.
        self.action_size: int = self.hand_size * (self.grid_size ** 2) * 2

        # Configure DQN hidden layers from AI_CONFIG.
        hidden_layers: tuple = AI_CONFIG.get("dqn_hidden_layers", (128, 128))
        self.policy_net = DQN(self.state_size, self.action_size, hidden_layers)
        self.target_net = DQN(self.state_size, self.action_size, hidden_layers)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=AI_CONFIG.get("lr", 0.001))
        self.epsilon: float = AI_CONFIG.get("epsilon", 0.1)  # Exploration rate

    def _get_board_state(self) -> np.ndarray:
        """
        Convert the current board (a dict mapping (x,y) to Card) into a fixed-size grid.
        The grid is 15x15, covering coordinates from -7 to +7 in both x and y.
        Each cell is encoded as:
          0: empty,
          1: start,
          2: path (includes all non-goal cards of type "path"),
          3: uncovered goal,
          4: hidden goal.
        
        Returns:
            np.ndarray: Flattened grid (length = 15*15).
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        offset = self.grid_size // 2  # 7 for grid_size 15
        for (x, y), card in self.env.board.items():
            grid_x = x + offset
            grid_y = y + offset
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                if card.type == "start":
                    grid[grid_y, grid_x] = 1
                elif card.type == "path":
                    grid[grid_y, grid_x] = 2
                elif card.type == "goal":
                    # For goal cards, differentiate uncovered vs hidden.
                    if card.hidden:
                        grid[grid_y, grid_x] = 4
                    else:
                        grid[grid_y, grid_x] = 3
        return grid.flatten()

    def _get_hand_state(self, player_index: int) -> np.ndarray:
        """
        Encode the player's hand as a vector of length hand_size.
        For each card, encode:
          1: start,
          2: path,
          3: uncovered goal,
          4: hidden goal.
        If the player has fewer than hand_size cards, pad with 0.

        Args:
            player_index (int): The index of the current player.

        Returns:
            np.ndarray: The hand encoding.
        """
        hand = self.env.player_hands[player_index]
        encoding = []
        for card in hand:
            if card.type == "start":
                encoding.append(1)
            elif card.type == "path":
                encoding.append(2)
            elif card.type == "goal":
                encoding.append(3 if not card.hidden else 4)
            else:
                encoding.append(0)
        # Pad if necessary.
        while len(encoding) < self.hand_size:
            encoding.append(0)
        return np.array(encoding, dtype=np.float32)

    def _get_state(self, player_index: int) -> np.ndarray:
        """
        Compute the full state representation for the current player.
        The state is the concatenation of the flattened board grid (15x15) and the hand encoding.

        Args:
            player_index (int): The index of the current player.

        Returns:
            np.ndarray: The state vector (length = 225 + hand_size).
        """
        board_state = self._get_board_state()
        hand_state = self._get_hand_state(player_index)
        return np.concatenate([board_state, hand_state])

    def _map_action(self, action_index: int) -> tuple[int, tuple[int, int], int]:
        """
        Map a discrete action index to a game action.
        The mapping is as follows:
          - card_index = action_index // (grid_size*grid_size*2)
          - remainder = action_index % (grid_size*grid_size*2)
          - x_index = remainder // (grid_size*2)
          - remainder2 = remainder % (grid_size*2)
          - y_index = remainder2 // 2
          - orientation = remainder2 % 2  (0 => 0°, 1 => 180°)
        The board cell (x_index, y_index) is mapped to board coordinates by subtracting (grid_size//2).
        
        Args:
            action_index (int): The chosen discrete action.
        
        Returns:
            tuple[int, tuple[int, int], int]: (card_index, board position, orientation)
        """
        H = self.hand_size
        N = self.grid_size  # 15
        total_per_card = N * N * 2
        card_index = action_index // total_per_card
        remainder = action_index % total_per_card
        x_index = remainder // (N * 2)
        remainder2 = remainder % (N * 2)
        y_index = remainder2 // 2
        orientation = remainder2 % 2
        # Map x_index, y_index (0..N-1) to board coordinates: subtract offset.
        offset = N // 2
        board_x = x_index - offset
        board_y = y_index - offset
        return (card_index, (board_x, board_y), orientation)

    def act(self, player_index: int) -> tuple[int, tuple[int, int], int]:
        """
        Choose an action using the DQN with epsilon-greedy policy.

        Args:
            player_index (int): The index of the current player.
        
        Returns:
            tuple[int, tuple[int, int], int]: The chosen game action.
        """
        state = self._get_state(player_index)
        # Use torch.from_numpy to convert state to tensor.
        state_tensor = torch.from_numpy(state).unsqueeze(0).float()
        if np.random.rand() < self.epsilon:
            # Random action
            action_index = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action_index = int(torch.argmax(q_values, dim=1).item())
        return self._map_action(action_index)
