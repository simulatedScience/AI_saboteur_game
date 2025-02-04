# base_agent.py
"""
This module defines AI agents for the Saboteur game.
Agents follow a common interface via the act() method.
"""

import random
from ..saboteur_env import SaboteurEnv
from ..cards import Card

class BaseAgent:
    def __init__(self, env: SaboteurEnv) -> None:
        """
        Initialize the BaseAgent.

        Args:
            env (SaboteurEnv): The game environment.
        """
        self.env = env
    
    def act(self, player_index: int) -> tuple[int, tuple[int, int], int]:
        """
        Choose an action for the given player.

        Args:
            player_index (int): The index of the current player.

        Raises:
            NotImplementedError: Subclasses must implement the act() method.

        Returns:
            tuple[int, tuple[int, int], int]: The chosen action.
        """
        raise NotImplementedError("Subclasses must implement the act() method.")
