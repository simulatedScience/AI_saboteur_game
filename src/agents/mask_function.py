# mask_function.py
"""
This module defines a mask function for SaboteurDiscreteWrapper.
The action space is MultiDiscrete([hand_size, 41, 41, 2]), whose flattened form has length 90.
The mask function returns a binary vector of length 90, where:
  - The first 6 entries correspond to card selection.
  - The next 41 entries correspond to the x coordinate.
  - The following 41 entries correspond to the y coordinate.
  - The last 2 entries correspond to the orientation.
An entry is 1 if there exists at least one valid move with that sub-action value.
"""
import numpy as np

from ..config import CONFIG, AI_CONFIG
from ..saboteur_env import SaboteurEnv, COORD_LOW, COORD_HIGH, COORD_RES

HAND_SIZE: int = CONFIG["hand_size"]

def mask_func(env: SaboteurEnv) -> np.ndarray:
    """
    Compute a binary mask over the flattened action space.
    
    The action space is MultiDiscrete([HAND_SIZE, COORD_RES, COORD_RES, 2]).
    With probability mask_dropout_prob, an unmasked (all ones) vector is returned.
    
    Args:
        env: The SaboteurDiscreteWrapper environment instance.
    
    Returns:
        np.ndarray: A binary mask of shape (HAND_SIZE + COORD_RES + COORD_RES + 2,).
    """
    # Mask dropout: Occasionally disable the mask.
    mask_dropout_prob: float = AI_CONFIG.get("mask_dropout_prob", 0.1)
    actions_dim: int = HAND_SIZE+1 + COORD_RES + COORD_RES + 2
    if np.random.rand() < mask_dropout_prob:
        return np.ones(actions_dim, dtype=np.uint8)
    # Initialize masks for card, x, y, and orientation.
    card_mask: np.ndarray = np.zeros(HAND_SIZE+1, dtype=np.uint8)
    x_mask: np.ndarray = np.zeros(COORD_RES, dtype=np.uint8)
    y_mask: np.ndarray = np.zeros(COORD_RES, dtype=np.uint8)
    orient_mask: np.ndarray = np.zeros(2, dtype=np.uint8)
    
    valid_moves: list[tuple[int, float, float, int]] = []
    for card_idx in range(HAND_SIZE):
        hand = env.player_hands[env.current_player]
        if card_idx >= len(hand):
            continue
        card = hand[card_idx]
        original_rotation: int = card.rotation
        for orientation in (0, 1):
            card.rotation = orientation
            valid_positions = env.get_valid_placements(card)
            if not valid_positions:
                continue
            for pos in valid_positions:
                valid_moves.append((card_idx, pos[0], pos[1], orientation))
        card.rotation = original_rotation
    
    if valid_moves:
        step_size: float = (COORD_HIGH - COORD_LOW) / (COORD_RES - 1)
        for move in valid_moves:
            c, x_val, y_val, o = move
            card_mask[c] = 1
            x_index: int = int(round((x_val - COORD_LOW) / step_size))
            y_index: int = int(round((y_val - COORD_LOW) / step_size))
            if 0 <= x_index < COORD_RES:
                x_mask[x_index] = 1
            if 0 <= y_index < COORD_RES:
                y_mask[y_index] = 1
            orient_mask[o] = 1
    return np.concatenate([card_mask, x_mask, y_mask, orient_mask])
