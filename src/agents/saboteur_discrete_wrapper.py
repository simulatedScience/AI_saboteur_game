# saboteur_discrete_wrapper.py
"""
SaboteurDiscreteWrapper wraps SaboteurEnv into a gym environment with a MultiDiscrete action space.
The observation is a fixed state vector of length 2332.
The action space is MultiDiscrete([hand_size, 41, 41, 2]). Discrete indices for x and y are mapped
into continuous coordinates and snapped to the nearest valid placement.
"""


import gymnasium as gym
import numpy as np
from ..saboteur_env import SaboteurEnv
from ..config import CONFIG

MAX_BOARD_CARDS: int = 100
CARD_FEATURES: int = 22
BOARD_STATE_SIZE: int = MAX_BOARD_CARDS * CARD_FEATURES
HAND_SIZE: int = CONFIG.get("hand_size", 6)
HAND_STATE_SIZE: int = HAND_SIZE * CARD_FEATURES
STATE_SIZE: int = BOARD_STATE_SIZE + HAND_STATE_SIZE

COORD_LOW: float = -10.0
COORD_HIGH: float = 10.0
COORD_RES: int = 41

class SaboteurDiscreteWrapper(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self) -> None:
        super().__init__()
        self.env = SaboteurEnv()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([HAND_SIZE, COORD_RES, COORD_RES, 2])
        self.step_size = (COORD_HIGH - COORD_LOW) / (COORD_RES - 1)
        
    def reset(self, **kwargs: any) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        state = self._get_state(self.env.current_player)
        return state, info
        
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        card_index = int(action[0])
        x_index = int(action[1])
        y_index = int(action[2])
        orientation = int(action[3])
        desired_x = COORD_LOW + x_index * self.step_size
        desired_y = COORD_LOW + y_index * self.step_size
        desired_coord = (desired_x, desired_y)
        hand = self.env.player_hands[self.env.current_player]
        if card_index >= len(hand):
            game_action = (-1, (0.0, 0.0), 0)
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
        self.env.render(mode)

    def _get_state(self, player_index: int) -> np.ndarray:
        board_state = self._encode_board()
        hand_state = self._encode_hand(player_index)
        return np.concatenate([board_state, hand_state])

    def _encode_board(self) -> np.ndarray:
        cards = list(self.env.board.values())
        cards.sort(key=lambda c: ((c.x if c.x is not None else 0), (c.y if c.y is not None else 0)))
        encodings = []
        for card in cards[:MAX_BOARD_CARDS]:
            encodings.append(self._encode_card(card))
        while len(encodings) < MAX_BOARD_CARDS:
            encodings.append(np.zeros(CARD_FEATURES, dtype=np.float32))
        return np.concatenate(encodings)
        
    def _encode_hand(self, player_index: int) -> np.ndarray:
        hand = self.env.player_hands[player_index]
        encodings = []
        for card in hand:
            encodings.append(self._encode_card(card))
        while len(encodings) < HAND_SIZE:
            encodings.append(np.zeros(CARD_FEATURES, dtype=np.float32))
        return np.concatenate(encodings)
        
    def _encode_card(self, card: any) -> np.ndarray:
        x = float(card.x) if card.x is not None else 0.0
        y = float(card.y) if card.y is not None else 0.0
        pos = [x, y]
        edges = []
        for edge in ["top", "right", "bottom", "left"]:
            etype = card.edges.get(edge, "wall")
            if etype == "wall":
                edges.extend([1, 0, 0])
            else:
                edges.extend([0, 1, 0])
        possible_connections = [("left","right"), ("left","top"), ("left","bottom"),
                                ("right","top"), ("right","bottom"), ("top","bottom")]
        conn = [1 if pair in card.connections else 0 for pair in possible_connections]
        hidden_goal = 1 if (card.type == "goal" and card.hidden) else 0
        start_flag = 1 if card.type == "start" else 0
        flags = [hidden_goal, start_flag]
        return np.array(pos + edges + conn + flags, dtype=np.float32)
        
    def _snap_to_valid(self, card: any, desired: tuple[float, float], orientation: int) -> tuple[float, float]:
        # Set card orientation
        card.rotation = orientation
        valid_positions = self.env.get_valid_placements(card)
        if not valid_positions:
            return (0.0, 0.0)
        distances = [np.linalg.norm(np.array(pos) - np.array(desired)) for pos in valid_positions]
        min_idx = int(np.argmin(distances))
        return valid_positions[min_idx]
