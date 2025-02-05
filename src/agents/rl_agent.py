# rl_agent.py
"""
This module implements an RL agent for Saboteur using a hybrid action space.
The agent follows the BaseAgent interface and outputs an action tuple:
    (card_index: int, (x: float, y: float), orientation: int)
The state is a 2332-dimensional vector encoding up to 100 board cards and 6 hand cards.
The policy network has three heads:
  - Card selection (discrete over 6 options)
  - Placement coordinates (continuous 2D output, to be snapped to the closest valid placement)
  - Orientation selection (discrete over 2 options)
The network architecture is configurable via AI_CONFIG.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent
from ..saboteur_env import SaboteurEnv
from ..config import AI_CONFIG, CONFIG

# Define state dimensions based on our documentation.
MAX_BOARD_CARDS: int = 100
CARD_FEATURES: int = 22  # per card
BOARD_STATE_SIZE: int = MAX_BOARD_CARDS * CARD_FEATURES  # 2200
HAND_SIZE: int = CONFIG.get("hand_size", 6)
HAND_STATE_SIZE: int = HAND_SIZE * CARD_FEATURES         # 132
STATE_SIZE: int = BOARD_STATE_SIZE + HAND_STATE_SIZE       # 2332

class RLPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple, hand_size: int) -> None:
        """
        Initialize the policy network.

        Args:
            input_dim (int): Dimension of the input state.
            hidden_layers (tuple): A tuple of integers specifying hidden layer sizes.
            hand_size (int): Number of cards in hand.
        """
        super().__init__()
        layers = []
        current_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())
            current_dim = h
        self.feature_extractor = nn.Sequential(*layers)
        # Head for card selection (6 outputs)
        self.card_head = nn.Linear(current_dim, hand_size)
        # Head for placement coordinates (2 outputs)
        self.coord_head = nn.Linear(current_dim, 2)
        # Head for orientation (2 outputs)
        self.orient_head = nn.Linear(current_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, input_dim).

        Returns:
            tuple: (card_logits, coord_output, orientation_logits)
        """
        features = self.feature_extractor(x)
        card_logits = self.card_head(features)
        coord_output = self.coord_head(features)
        orientation_logits = self.orient_head(features)
        return card_logits, coord_output, orientation_logits

class RLAgent(BaseAgent):
    def __init__(self, env: SaboteurEnv) -> None:
        """
        Initialize the RLAgent.

        Args:
            env (SaboteurEnv): The Saboteur game environment.
        """
        super().__init__(env)
        # Set state and action sizes based on design.
        self.state_size: int = STATE_SIZE
        # The action output is a tuple: card_index (6), continuous (x, y) and orientation (2).
        self.hand_size: int = CONFIG.get("hand_size", 6)
        self.action_space_dims = {
            "card": self.hand_size,
            "coord": 2,    # continuous output; bounds [-10, 10]
            "orient": 2
        }
        hidden_layers: tuple = AI_CONFIG.get("dqn_hidden_layers", (256, 256))
        self.policy_net = RLPolicy(self.state_size, hidden_layers, self.hand_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=AI_CONFIG.get("lr", 0.001))
        self.epsilon: float = AI_CONFIG.get("epsilon", 0.1)

    def _get_state(self, player_index: int) -> np.ndarray:
        """
        Generate the state vector for the given player.
        The state is the concatenation of:
          - Board state: fixed-size encoding of up to 100 board cards.
          - Hand state: encoding of the current player's hand (6 cards).
        
        Args:
            player_index (int): Index of the current player.
        
        Returns:
            np.ndarray: State vector of shape (STATE_SIZE,).
        """
        # Encode board state: sort env.board by (x,y), encode each card using its features.
        board_cards = list(self.env.board.values())
        board_cards.sort(key=lambda c: ((c.x if c.x is not None else 0), (c.y if c.y is not None else 0)))
        board_encodings = []
        # Assume we have a function to encode a card into a vector of length CARD_FEATURES.
        # For this example, we implement a simple encoder inline.
        def encode_card(card) -> np.ndarray:
            # Position (x, y) as floats (if None, use 0). We do not normalize here.
            x = card.x if card.x is not None else 0.0
            y = card.y if card.y is not None else 0.0
            pos = [x, y]
            # Edge types: for each edge, one-hot encoding for [wall, path, dead-end]
            edge_order = ["top", "right", "bottom", "left"]
            edge_enc = []
            for edge in edge_order:
                if card.edges.get(edge, "wall") == "wall":
                    edge_enc.extend([1, 0, 0])
                elif card.edges.get(edge, "wall") in ("path", "dead-end"):
                    # Use [0,1,0] for path and [0,1,0] for dead-end (treat them similarly) 
                    edge_enc.extend([0, 1, 0])
                else:
                    edge_enc.extend([1, 0, 0])
            # Connection info: 6 binary values for fixed pairs (e.g., sorted lexicographically)
            possible_connections = [("left","right"), ("left","top"), ("left","bottom"),
                                    ("right","top"), ("right","bottom"), ("top","bottom")]
            conn_enc = [1 if pair in card.connections else 0 for pair in possible_connections]
            # Special flags: hidden goal flag and start flag.
            hidden_goal = 1 if (card.type == "goal" and card.hidden) else 0
            start_flag = 1 if card.type == "start" else 0
            flags = [hidden_goal, start_flag]
            return np.array(pos + edge_enc + conn_enc + flags, dtype=np.float32)
        for card in board_cards[:MAX_BOARD_CARDS]:
            board_encodings.append(encode_card(card))
        # Pad if necessary.
        while len(board_encodings) < MAX_BOARD_CARDS:
            board_encodings.append(np.zeros(CARD_FEATURES, dtype=np.float32))
        board_state = np.concatenate(board_encodings)
        # Encode hand state.
        hand = self.env.player_hands[player_index]
        hand_encodings = []
        for card in hand:
            hand_encodings.append(encode_card(card))
        while len(hand_encodings) < self.hand_size:
            hand_encodings.append(np.zeros(CARD_FEATURES, dtype=np.float32))
        hand_state = np.concatenate(hand_encodings)
        return np.concatenate([board_state, hand_state])

    def _snap_to_valid(self, card: any, desired: tuple[float, float], orientation: int) -> tuple[float, float]:
        """
        Given a card and a desired (x, y) continuous output, snap to the closest valid placement.
        Uses env.get_valid_placements(card) to get a list of valid positions.
        If none are available, returns (0.0, 0.0) (which signals a skip).
        
        Args:
            card: The card object.
            desired (tuple[float, float]): Desired (x, y) output from the network.
            orientation (int): The chosen orientation.
        
        Returns:
            tuple[float, float]: The valid placement (snapped) coordinates.
        """
        valid_positions = self.env.get_valid_placements(card)
        if not valid_positions:
            return (0.0, 0.0)
        # Compute Euclidean distance to desired point.
        distances = [np.linalg.norm(np.array(pos) - np.array(desired)) for pos in valid_positions]
        min_idx = int(np.argmin(distances))
        return valid_positions[min_idx]

    def act(self, player_index: int) -> tuple[int, tuple[float, float], int]:
        """
        Select an action using the policy network with epsilon-greedy exploration.
        The network outputs:
          - card_logits: distribution over hand cards.
          - coord_output: continuous (x, y) output.
          - orientation_logits: distribution over orientations.
        The chosen continuous coordinate is snapped to the nearest valid placement for the chosen card.
        If no valid placement exists, returns a skip action (-1, (0.0, 0.0), 0).
        
        Args:
            player_index (int): The index of the current player.
        
        Returns:
            tuple[int, tuple[float, float], int]: (card_index, (x, y), orientation)
        """
        state = self._get_state(player_index)
        state_tensor = torch.from_numpy(state).unsqueeze(0).float()
        # Epsilon-greedy exploration.
        if random.random() < self.epsilon:
            card_index = random.randrange(self.hand_size)
            # For continuous part, sample uniformly in [-10, 10].
            coord = (random.uniform(-10, 10), random.uniform(-10, 10))
            orientation = random.randrange(2)
        else:
            with torch.no_grad():
                card_logits, coord_output, orient_logits = self.policy_net(state_tensor)
            card_index = int(torch.argmax(card_logits, dim=1).item())
            # coord_output is a 2-dim tensor.
            coord = tuple(coord_output.squeeze(0).tolist())
            orientation = int(torch.argmax(orient_logits, dim=1).item())
        # Get the selected card.
        hand = self.env.player_hands[player_index]
        if card_index >= len(hand):
            # Fallback: skip action.
            return (-1, (0.0, 0.0), 0)
        selected_card = hand[card_index]
        # Snap the continuous coordinate to the closest valid placement.
        snapped_coord = self._snap_to_valid(selected_card, coord, orientation)
        # If no valid placement found, return skip.
        if snapped_coord == (0.0, 0.0):
            return (-1, (0.0, 0.0), 0)
        return (card_index, snapped_coord, orientation)
