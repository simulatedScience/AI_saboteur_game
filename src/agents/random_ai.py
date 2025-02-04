# random_ai.py

import random
from ..saboteur_env import SaboteurEnv
from ..cards import Card

class RandomAgent:
    def __init__(self, env: SaboteurEnv) -> None:
        """
        Initialize the RandomAgent.

        Args:
            env (SaboteurEnv): The game environment.
        """
        self.env = env

    def act(self, player_index: int) -> tuple[int, tuple[int, int], int]:
        """
        Choose a random valid action from the player's hand.
        If no valid actions exist, returns a skip action.

        Args:
            player_index (int): The index of the current player.

        Returns:
            tuple: (card_index, board position, orientation)
        """
        hand = self.env.player_hands[player_index]
        valid_actions: list[tuple[int, tuple[int, int], int]] = []
        for i, card in enumerate(hand):
            placements = self.env.get_valid_placements(card)
            for pos in placements:
                valid_actions.append((i, pos, card.rotation))
        if valid_actions:
            return random.choice(valid_actions)
        else:
            return (-1, (0, 0), 0)