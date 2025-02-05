# saboteur_hybrid_wrapper.py
"""
SaboteurHybridWrapper wraps the SaboteurEnv into a gym environment with a hybrid
action space. The observation is a flat state vector (of fixed size STATE_SIZE) that
encodes the board (up to 100 cards, each with 22 features) and the current player's hand
(6 cards, each with 22 features). The resulting state vector length is 2332.

The action space is a gym.spaces.Dict with:
  - "card": Discrete (0 to hand_size-1)
  - "coord": Box(low=-10, high=10, shape=(2,), dtype=np.float32)
  - "orient": Discrete (0 or 1)

The step() method converts the dictionary action into the game action tuple
(card_index, snapped (x, y), orientation), where the continuous coordinate is snapped
to the nearest valid placement.
"""

import gymnasium as gym
import numpy as np

from ..saboteur_env import SaboteurEnv
from ..config import CONFIG

# Fixed parameters (as described in our RL documentation)
MAX_BOARD_CARDS: int = 100
CARD_FEATURES: int = 22  # per card
BOARD_STATE_SIZE: int = MAX_BOARD_CARDS * CARD_FEATURES  # 2200
HAND_SIZE: int = CONFIG.get("hand_size", 6)
HAND_STATE_SIZE: int = HAND_SIZE * CARD_FEATURES         # 132
STATE_SIZE: int = BOARD_STATE_SIZE + HAND_STATE_SIZE       # 2332

class SaboteurHybridWrapper(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self) -> None:
        """
        Initialize the SaboteurHybridWrapper.
        """
        super().__init__()
        self.env = SaboteurEnv()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            "card": gym.spaces.Discrete(HAND_SIZE),
            "coord": gym.spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            "orient": gym.spaces.Discrete(2)
        })

    def reset(self, **kwargs: any) -> tuple[np.ndarray, dict]:
        """
        Reset the underlying environment and return the initial state.

        Returns:
            tuple[np.ndarray, dict]: (state vector, info dictionary)
        """
        obs, info = self.env.reset(**kwargs)
        state = self._get_state(self.env.current_player)
        return state, info

    def step(self, action: dict) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment using the dictionary action.
        
        Args:
            action (dict): Dictionary with keys "card", "coord", "orient".
            
        Returns:
            tuple: (state, reward, done, truncated, info)
        """
        card_index: int = action["card"]
        desired_coord: tuple[float, float] = tuple(action["coord"])
        orientation: int = action["orient"]
        # Get the selected card from the current player's hand.
        hand = self.env.player_hands[self.env.current_player]
        if card_index >= len(hand):
            game_action = (-1, (0.0, 0.0), 0)  # Skip action
        else:
            selected_card = hand[card_index]
            snapped_coord = self._snap_to_valid(selected_card, desired_coord, orientation)
            if snapped_coord == (0.0, 0.0):
                game_action = (-1, (0.0, 0.0), 0)
            else:
                game_action = (card_index, snapped_coord, orientation)
        obs, reward, done, truncated, info = self.env.step(game_action)
        state = self._get_state(self.env.current_player)
        return state, reward, done, truncated, info

    def render(self, mode: str = "human") -> None:
        """
        Render the underlying environment.
        """
        self.env.render(mode)

    def _get_state(self, player_index: int) -> np.ndarray:
        """
        Build the state vector by concatenating the board and hand encodings.
        For each card, we encode 22 features (see RL documentation).
        
        Args:
            player_index (int): Index of the current player.
            
        Returns:
            np.ndarray: State vector of length STATE_SIZE.
        """
        # Encode board: sort cards by (x,y) and encode each card.
        board_cards = list(self.env.board.values())
        board_cards.sort(key=lambda c: ((c.x if c.x is not None else 0), (c.y if c.y is not None else 0)))
        board_encodings = []
        for card in board_cards[:MAX_BOARD_CARDS]:
            board_encodings.append(self._encode_card(card))
        while len(board_encodings) < MAX_BOARD_CARDS:
            board_encodings.append(np.zeros(CARD_FEATURES, dtype=np.float32))
        board_state = np.concatenate(board_encodings)
        # Encode hand.
        hand = self.env.player_hands[player_index]
        hand_encodings = []
        for card in hand:
            hand_encodings.append(self._encode_card(card))
        while len(hand_encodings) < HAND_SIZE:
            hand_encodings.append(np.zeros(CARD_FEATURES, dtype=np.float32))
        hand_state = np.concatenate(hand_encodings)
        return np.concatenate([board_state, hand_state])

    def _encode_card(self, card: any) -> np.ndarray:
        """
        Encode a card into a 22-dimensional feature vector.
        Features:
          - Position (x, y): if None, use 0; as floats.
          - Edge types for top, right, bottom, left: one-hot for [wall, path, dead-end] (3 values per edge, total 12).
          - Connection bits: 6 binary values for fixed ordered pairs.
          - Special flags: hidden goal flag (1 if hidden goal, else 0) and start flag (1 if start, else 0).
        
        Args:
            card: A Card object.
        
        Returns:
            np.ndarray: A vector of length 22.
        """
        # Position.
        x = float(card.x) if card.x is not None else 0.0
        y = float(card.y) if card.y is not None else 0.0
        pos = [x, y]
        # Edge types.
        edges = []
        for edge in ["top", "right", "bottom", "left"]:
            etype = card.edges.get(edge, "wall")
            if etype == "wall":
                edges.extend([1, 0, 0])
            else:  # treat both "path" and "dead-end" as path for placement
                edges.extend([0, 1, 0])
        # Connection bits.
        possible_connections = [("left","right"), ("left","top"), ("left","bottom"),
                                ("right","top"), ("right","bottom"), ("top","bottom")]
        conn = [1 if pair in card.connections else 0 for pair in possible_connections]
        # Special flags.
        hidden_goal = 1 if (card.type == "goal" and card.hidden) else 0
        start_flag = 1 if card.type == "start" else 0
        flags = [hidden_goal, start_flag]
        return np.array(pos + edges + conn + flags, dtype=np.float32)

    def _snap_to_valid(self, card: any, desired: tuple[float, float], orientation: int) -> tuple[float, float]:
        """
        Snap the desired continuous coordinate to the nearest valid placement for the card.
        
        Args:
            card: The card object.
            desired (tuple[float, float]): Desired (x, y) coordinate.
            orientation (int): Orientation (0 or 1).
        
        Returns:
            tuple[float, float]: The valid placement coordinate; (0.0, 0.0) if none available.
        """
        valid_positions = self.env.get_valid_placements(card)
        if not valid_positions:
            return (0.0, 0.0)
        distances = [np.linalg.norm(np.array(pos) - np.array(desired)) for pos in valid_positions]
        min_idx = int(np.argmin(distances))
        return valid_positions[min_idx]
