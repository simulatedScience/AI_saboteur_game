# game_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .cards import create_game_board
from .config import config

class SaboteurEnv(gym.Env):
    """
    A minimal Gymnasium environment for the Saboteur card game.
    This version uses a game board with a start card and three goal cards.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players=None):
        super().__init__()
        self.num_players = num_players if num_players is not None else config['numPlayers']

        # For now, assume a discrete action space with placeholder actions.
        self.action_space = spaces.Discrete(10)  # Placeholder for actual actions.

        # Observation: we represent the board as a matrix.
        # We'll encode card types as numbers:
        # 0: dead-end, 1: path, 2: start, 3: gold (if revealed), 4: coal (if revealed), 5: hidden goal.
        board_shape = (config['boardRows'], config['boardCols'])
        self.observation_space = spaces.Box(low=0, high=5, shape=board_shape, dtype=np.int8)

        self.reset()

    def reset(self):
        """Reset the environment state and return the initial observation."""
        self.board = create_game_board()
        self.current_player = 0
        self.done = False
        self.info = {}
        return self._get_obs(), {}

    def step(self, action):
        """
        Process an action and update the game state.
        This is highly simplified. In a full implementation, 'action'
        would determine a card play, a communication move, etc.
        """
        # (Placeholder) Process the action.
        # For now, simply cycle to the next player.
        reward = 0
        self.current_player = (self.current_player + 1) % self.num_players

        # If a terminal condition is reached, set self.done = True.
        obs = self._get_obs()
        return obs, reward, self.done, False, self.info

    def render(self, mode='human'):
        """
        Render the board to the console.
        A GUI-based render is provided in gui.py.
        """
        board_matrix = self._get_obs()
        for row in board_matrix:
            print(" ".join(str(cell) for cell in row))
        print()

    def _get_obs(self):
        """
        Convert the board state (list of Card objects) into a matrix.
        Mapping:
          - 'dead-end' -> 0
          - 'path'     -> 1
          - 'start'    -> 2
          - 'gold':   revealed -> 3, hidden -> 5
          - 'coal':   revealed -> 4, hidden -> 5
        """
        obs = np.zeros((config['boardRows'], config['boardCols']), dtype=np.int8)
        for card in self.board:
            if card.type == 'dead-end':
                obs[card.y_index, card.x_index] = 0
            elif card.type == 'path':
                obs[card.y_index, card.x_index] = 1
            elif card.type == 'start':
                obs[card.y_index, card.x_index] = 2
            elif card.type in ['gold', 'coal']:
                # Use 3 or 4 if revealed, otherwise 5 for hidden goal card.
                if card.hidden:
                    obs[card.y_index, card.x_index] = 5
                else:
                    obs[card.y_index, card.x_index] = 3 if card.type == 'gold' else 4
        return obs