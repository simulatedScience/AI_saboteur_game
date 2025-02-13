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
        return self.env.reset(**kwargs)
        
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        card_index = int(action[0])
        x_index = int(action[1])
        y_index = int(action[2])
        orientation = int(action[3])
        desired_x = COORD_LOW + x_index * self.step_size
        desired_y = COORD_LOW + y_index * self.step_size
        desired_coord = (desired_x, desired_y)
        hand = self.env.player_hands[self.env.current_player]
        # Check if the card index is valid
        if card_index >= len(hand):
            game_action = (-1, (0.0, 0.0), 0)
        else:
            # Get the selected card
            selected_card = hand[card_index]
            # apply the rotation
            if selected_card.rotation != orientation:
                selected_card.rotate()
            # Snap the desired coordinate to the nearest valid placement, keeping the orientation
            snapped_coord = self._snap_to_valid(selected_card, desired_coord, orientation)
            if snapped_coord == (0.0, 0.0):
                game_action = (-1, (0.0, 0.0), 0)
            else:
                game_action = (card_index, snapped_coord, orientation)

        obs, reward, done, truncated, info = self.env.step(game_action)
        return obs, reward, done, truncated, info

    def render(self, mode: str = "human") -> None:
        self.env.render(mode)

    def _snap_to_valid(self, card: any, desired: tuple[float, float], orientation: int) -> tuple[float, float]:
        # Set card orientation
        card.rotation = orientation
        valid_positions = self.env.get_valid_placements(card)
        if not valid_positions:
            return (0.0, 0.0)
        distances = [np.linalg.norm(np.array(pos) - np.array(desired)) for pos in valid_positions]
        min_idx = int(np.argmin(distances))
        return valid_positions[min_idx]
