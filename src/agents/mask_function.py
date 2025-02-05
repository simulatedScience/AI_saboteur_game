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

from ..config import CONFIG
from ..saboteur_env import SaboteurEnv
from .saboteur_discrete_wrapper import COORD_LOW, COORD_HIGH, COORD_RES

HAND_SIZE: int = CONFIG.get("hand_size", 6)
# Sub-action dimensions:
card_dim = HAND_SIZE           # 6
x_dim = COORD_RES              # 41
y_dim = COORD_RES              # 41
orient_dim = 2                 # 2

def mask_func(env: SaboteurEnv) -> np.ndarray:
    """
    Compute a binary mask over the flattened action space.
    
    For each sub-action dimension, we determine which discrete options appear in at least one valid move.
    We then flatten the per-dimension masks into a single vector of length (6+41+41+2)=90.
    
    Args:
        env: The SaboteurDiscreteWrapper environment.
        obs: The observation (not used).
        
    Returns:
        np.ndarray: A binary mask of shape (90,), where each segment corresponds to a sub-action dimension.
    """
    # Initialize per-dimension masks.
    card_mask = np.zeros(card_dim, dtype=np.uint8)
    x_mask = np.zeros(x_dim, dtype=np.uint8)
    y_mask = np.zeros(y_dim, dtype=np.uint8)
    orient_mask = np.zeros(orient_dim, dtype=np.uint8)
    
    # Collect valid moves: a valid move is (card_index, (x, y), orientation)
    valid_moves = []
    for card_idx in range(card_dim):
        hand = env.env.player_hands[env.env.current_player]
        if card_idx >= len(hand):
            continue
        card: "Card" = hand[card_idx]
        original_rotation: int = card.rotation
        for orientation in (0, 1):
            # set card orientation
            card.rotation = orientation
            valid_positions = env.env.get_valid_placements(card)
            if not valid_positions:
                continue
            for pos in valid_positions:
                valid_moves.append((card_idx, pos[0], pos[1], orientation))
        # restore original rotation
        card.rotation = original_rotation

    if valid_moves:
        # For each valid move, map continuous coordinates to discrete indices.
        step_size = (COORD_HIGH - COORD_LOW) / (COORD_RES - 1)
        for move in valid_moves:
            c, x_val, y_val, o = move
            card_mask[c] = 1
            # Map x_val, y_val to discrete indices.
            x_index = int(round((x_val - COORD_LOW) / step_size))
            y_index = int(round((y_val - COORD_LOW) / step_size))
            if 0 <= x_index < x_dim:
                x_mask[x_index] = 1
            if 0 <= y_index < y_dim:
                y_mask[y_index] = 1
            orient_mask[o] = 1
    # Flatten the masks by concatenating.
    flat_mask = np.concatenate([card_mask, x_mask, y_mask, orient_mask])
    return flat_mask
