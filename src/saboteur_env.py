# saboteur_env.py
"""
Gymnasium environment for the Saboteur card game.
Creates an almost empty board with four cards:
  - A 4-way start tile at (0, 0)
  - Three goal cards at (8, 0), (8, 2), and (8, -2)
Also maintains a placeholder for a path connectivity graph.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict
from .config import CONFIG
from .cards import Card
import random

class SaboteurEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players: int = None) -> None:
        """
        Initialize the Saboteur environment.
        
        Args:
            num_players (Optional[int]): Number of players (default from CONFIG).
        """
        super().__init__()
        self.num_players: int = num_players if num_players is not None else CONFIG['numPlayers']
        # Represent the board as a dictionary mapping (x, y) coordinates to Cards.
        self.board: Dict[Tuple[int, int], Card] = {}
        # Placeholder for a graph structure to track built paths.
        self.path_graph: Dict[Tuple[int, int], Dict[str, str]] = {}
        self._create_initial_board()
        # Dummy action and observation spaces.
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)
        self.current_player: int = 0
        self.done: bool = False
        self.info: dict = {}

    def _create_initial_board(self) -> None:
        """
        Create an almost empty board with only four cards:
            - A 4-way start tile at (0, 0)
            - Three goal cards at (8, 0), (8, 2), and (8, -2)
        """
        # Create the start tile at (0, 0): all edges are 'path'.
        start_edges = {'top': 'path', 'right': 'path', 'bottom': 'path', 'left': 'path'}
        start_connections = [
            ('top', 'right'), ('top', 'bottom'), ('top', 'left'),
            ('right', 'bottom'), ('right', 'left'), ('bottom', 'left')
        ]
        start_card = Card('start', x=0, y=0, edges=start_edges, connections=start_connections)
        self.board[(0, 0)] = start_card
        self.path_graph[(0, 0)] = start_card.edges  # (Placeholder)

        # Define goal positions.
        goal_positions = [(8, 0), (8, 2), (8, -2)]
        gold_index = random.randint(0, len(goal_positions) - 1)
        for idx, pos in enumerate(goal_positions):
            # For goal cards, use all dead-end edges.
            goal_edges = {'top': 'dead-end', 'right': 'dead-end', 'bottom': 'dead-end', 'left': 'dead-end'}
            goal_card = Card('goal', x=pos[0], y=pos[1], edges=goal_edges, connections=[], 
                             goal_type='gold' if idx == gold_index else 'coal')
            goal_card.hidden = True
            self.board[pos] = goal_card

    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment and return a dummy observation.
        """
        self.board = {}
        self.path_graph = {}
        self._create_initial_board()
        self.current_player = 0
        self.done = False
        self.info = {}
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, dict]:
        """
        Process an action (placeholder) and update the game state.
        
        Args:
            action (int): The action to process.
        
        Returns:
            A tuple of (observation, reward, done, truncated, info).
        """
        reward: int = 0
        self.current_player = (self.current_player + 1) % self.num_players
        return self._get_obs(), reward, self.done, False, self.info

    def _get_obs(self) -> np.ndarray:
        """
        Return a dummy observation.
        """
        return np.array([0])

    def render(self, mode: str = 'human') -> None:
        """
        Render the board state to the console for debugging.
        """
        print("Board:")
        for pos, card in self.board.items():
            print(f"Position {pos}: {card}")
