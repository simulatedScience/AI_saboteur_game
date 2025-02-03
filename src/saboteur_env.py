# saboteur_env.py
"""
Updated SaboteurEnv implementation with card placement rules.

Features implemented:
1. A new method place_card(...) that enforces valid card placement.
   - The new card must be adjacent to an existing card.
   - The edges must align properly with neighbors.
   - There must be a continuous path from the start tile to the newly placed card (i.e. BFS or DFS from the start tile must be able to reach the new position after placement).
"""

# Standard library imports
import random
from collections import deque

# Third-party imports
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Local imports
from .config import CONFIG
from .cards import Card, load_deck


def edges_line_up(card_a: Card, card_b: Card, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> bool:
    """
    Check if the edges between card_a and card_b line up appropriately.
    That is, the edges are not walls in either direction and they match.
    """
    ax, ay = pos_a
    bx, by = pos_b
    dx = bx - ax
    dy = by - ay

    edge_a = None
    edge_b = None
    if dx == 1 and dy == 0:
        edge_a = card_a.edges['right']
        edge_b = card_b.edges['left']
    elif dx == -1 and dy == 0:
        edge_a = card_a.edges['left']
        edge_b = card_b.edges['right']
    elif dx == 0 and dy == 1:
        edge_a = card_a.edges['bottom']
        edge_b = card_b.edges['top']
    elif dx == 0 and dy == -1:
        edge_a = card_a.edges['top']
        edge_b = card_b.edges['bottom']
    else:
        return False

    if edge_a == 'wall' or edge_b == 'wall':
        return False

    return True


class SaboteurEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players: int | None = None) -> None:
        """
        Initialize the Saboteur environment.

        Args:
            num_players (int | None): Number of players (default from CONFIG).
        """
        super().__init__()
        self.num_players: int = num_players if num_players is not None else CONFIG['num_players']
        self.board: dict[tuple[int, int], Card] = {}
        self.start_position: tuple[int, int] = (0, 0)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)
        self.done: bool = False
        self.info: dict = {}
        self._create_initial_board()
        self.current_player: int = 0

        # load deck and deal hands
        self.deck = load_deck(deck_config_path=CONFIG['deck'])
        self.player_hands: dict[int, list[Card]] = {
            player_idx: [self.deck.pop() for _ in range(CONFIG['hand_size'])]
            for player_idx in range(self.num_players)
        }

    def _create_initial_board(self) -> None:
        """
        Create an almost empty board with only four cards:
         - A 4-way start tile at (0, 0)
         - Three goal cards at (8, 0), (8, 2), and (8, -2)
        """
        self.board.clear()
        start_edges = {
            'top': 'path',
            'right': 'path',
            'bottom': 'path',
            'left': 'path'
        }
        start_card = Card('start', x=0, y=0, edges=start_edges)
        self.board[(0, 0)] = start_card
        self.start_position = (0, 0)

        goal_positions = [(8, 0), (8, 2), (8, -2)]
        gold_index = random.randint(0, len(goal_positions) - 1)
        for idx, pos in enumerate(goal_positions):
            coal_edges = {
                'top': 'wall',
                'right': 'wall',
                'bottom': 'path',
                'left': 'path'
            }
            goal_card = Card(
                'goal',
                x=pos[0],
                y=pos[1],
                edges=start_edges if idx == gold_index else coal_edges,
                connections=[],
                goal_type='gold' if idx == gold_index else 'coal'
            )
            goal_card.hidden = True
            self.board[pos] = goal_card

    def reset(self) -> tuple[np.ndarray, dict]:
        """
        Reset the environment and return a dummy observation.
        """
        self._create_initial_board()
        self.current_player = 0
        self.done = False
        self.info = {}
        self.deck = load_deck(deck_config_path=CONFIG['deck'])
        self.player_hands = {
            player_idx: [self.deck.pop() for _ in range(CONFIG['hand_size'])]
            for player_idx in range(self.num_players)
        }
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        Return a dummy observation.
        """
        return np.array([0])

    def step(self, action: tuple[int, tuple[int, int], int]) -> tuple[np.ndarray, int, bool, bool, dict]:
        """
        Process an action consisting of:
          - card index in current player's hand,
          - board position where the card is to be placed, and
          - desired card orientation (0 or 180).
        Returns:
            observation, reward, done flag, truncated flag, info dict.
        """
        current_player: int = self.current_player
        current_hand: list[Card] = self.player_hands[current_player]
        card_index, pos, orientation = action

        # Validate card index
        if card_index < 0 or card_index >= len(current_hand):
            print("Invalid card index!")
            return self._get_obs(), -1, False, False, self.info

        played_card: Card = current_hand[card_index]
        # Set card orientation as needed (cards only have 0 or 180°)
        if played_card.rotation != orientation:
            played_card.rotate()

        # Attempt to place the card on the board
        success: bool = self.place_card(played_card, pos)
        if not success:
            print("Invalid card placement!")
            return self._get_obs(), -1, False, False, self.info

        # Remove the card from the player's hand
        del current_hand[card_index]

        # Draw a new card for the current player if available
        if self.deck:
            new_card: Card = self.deck.pop()
            current_hand.append(new_card)

        # Check termination conditions:
        # Condition 1: all player hands are empty.
        all_hands_empty: bool = all(len(hand) == 0 for hand in self.player_hands.values())
        # Condition 2: the gold goal card is revealed and reachable.
        gold_reached: bool = False
        for board_pos, card in self.board.items():
            if card.type == 'goal' and card.goal_type == 'gold' and not card.hidden:
                if self.can_reach(board_pos):
                    gold_reached = True
                    break

        if all_hands_empty or gold_reached:
            self.done = True
        else:
            self.done = False
            # Switch to the next player only if the game is not over.
            self.current_player = (self.current_player + 1) % self.num_players

        return self._get_obs(), 0, self.done, False, self.info

    def render(self, mode: str = 'human') -> None:
        """
        Render the board state to the console for debugging.
        """
        print("Board:")
        for pos, card in self.board.items():
            print(f"Position {pos}: {card}")
        print(f"Current player: {self.current_player}")
        for player, hand in self.player_hands.items():
            print(f"Player {player} hand: {hand}")

    def is_adjacent_to_board(self, pos: tuple[int, int]) -> bool:
        """
        Check if pos is adjacent to at least one existing board position.
        """
        x, y = pos
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return any(n in self.board for n in neighbors)

    def place_card(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Attempt to place the given card at pos on the board.
        Return True if the placement is valid and performed, False otherwise.
        """
        if pos in self.board:
            return False
        if not self.is_adjacent_to_board(pos):
            return False

        x, y = pos
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for npos in neighbors:
            if npos in self.board:
                neighbor_card = self.board[npos]
                if not edges_line_up(card, neighbor_card, pos, npos):
                    return False

        card.x, card.y = pos
        self.board[pos] = card

        if not self.can_reach(pos):
            del self.board[pos]
            return False

        return True

    def can_reach(self, target_pos: tuple[int, int]) -> bool:
        """
        Check if there's a path from self.start_position to target_pos using BFS.
        """
        if self.start_position not in self.board or target_pos not in self.board:
            return False

        queue = deque()
        visited = set()
        queue.append(self.start_position)
        visited.add(self.start_position)

        while queue:
            current = queue.popleft()
            if current == target_pos:
                return True

            cx, cy = current
            neighbors = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
            for npos in neighbors:
                if npos not in visited and npos in self.board:
                    if edges_line_up(self.board[current], self.board[npos], current, npos):
                        visited.add(npos)
                        queue.append(npos)
        return False


if __name__ == "__main__":
    env = SaboteurEnv()
    env.reset()
    env.render()