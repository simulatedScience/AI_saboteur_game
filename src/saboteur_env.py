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


def edges_line_up(card_a: Card, card_b: Card, pos_a: tuple[int,int], pos_b: tuple[int,int]) -> bool:
    """
    Check if the edges between card_a and card_b line up appropriately.
    That is, the edges are not walls in both directions, and they match.

    In standard Saboteur, we consider that an edge from A to B must not be 'wall' if B's corresponding edge is also not 'wall'.
    Typically we allow 'path' <-> 'path' or 'dead-end' <-> 'dead-end' or 'dead-end' <-> 'path'. We just disallow 'wall'.

    pos_a and pos_b are used to figure out which edges to compare.
    """
    ax, ay = pos_a
    bx, by = pos_b
    dx = bx - ax
    dy = by - ay

    # Figure out which edges we compare.
    # If dx = 1 => b is right of a => compare card_a.edges['right'] and card_b.edges['left'].
    # If dx = -1 => b is left of a => compare card_a.edges['left'] and card_b.edges['right'].
    # If dy = 1 => b is below a => compare card_a.edges['bottom'] and card_b.edges['top'].
    # If dy = -1 => b is above a => compare card_a.edges['top'] and card_b.edges['bottom'].

    edge_a = None
    edge_b = None
    if dx == 1 and dy == 0:
        # b is right of a
        edge_a = card_a.edges['right']
        edge_b = card_b.edges['left']
    elif dx == -1 and dy == 0:
        # b is left of a
        edge_a = card_a.edges['left']
        edge_b = card_b.edges['right']
    elif dx == 0 and dy == 1:
        # b is below a
        edge_a = card_a.edges['bottom']
        edge_b = card_b.edges['top']
    elif dx == 0 and dy == -1:
        # b is above a
        edge_a = card_a.edges['top']
        edge_b = card_b.edges['bottom']
    else:
        # Not adjacent
        return False

    # If either side is 'wall', that means no connection.
    # For valid adjacency, if one side is 'path' or 'dead-end', the other side must not be 'wall'.
    if edge_a == 'wall' or edge_b == 'wall':
        # If both are 'wall', it doesn't matter, there's no connection.
        # Actually in Saboteur, we typically don't want to place next to a wall if the neighbor is a card, unless both are walls?
        # But we'll interpret that we want them to line up as a path if it's a path or dead-end.
        # We say it's not a valid connection if either is 'wall'.
        return False

    # If both edges are not 'wall', we consider them lined up. We do not strictly require 'path' <-> 'path'.
    # Because in Saboteur, 'dead-end' is effectively a path end, so we allow 'dead-end' <-> 'path' or 'dead-end' <-> 'dead-end'.
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
        # Represent the board as a dictionary mapping (x, y) coordinates to Cards.
        self.board: dict[tuple[int, int], Card] = {}
        # We'll store the location of the start tile(s). For now, we assume there's just one at (0, 0)
        self.start_position: tuple[int, int] = (0, 0)

        # Dummy action and observation spaces.
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)
        # initial setup
        self.reset()

    def _create_initial_board(self) -> None:
        """
        Create an almost empty board with only four cards:
          - A 4-way start tile at (0, 0)
          - Three goal cards at (8, 0), (8, 2), and (8, -2)
        """
        self.board.clear()
        # Create the start tile at (0, 0): all edges are 'path'.
        start_edges = {
            'top': 'path',
            'right': 'path',
            'bottom': 'path',
            'left': 'path'
        }
        start_card = Card('start', x=0, y=0, edges=start_edges)
        self.board[(0, 0)] = start_card
        self.start_position = (0, 0)

        # Define goal positions.
        goal_positions = [(8, 0), (8, 2), (8, -2)]
        gold_index = random.randint(0, len(goal_positions) - 1)
        for idx, pos in enumerate(goal_positions):
            # For goal cards, use all dead-end edges.
            coal_edges = {
                'top': 'wall',
                'right': 'wall',
                'bottom': 'path',
                'left': 'path'}
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

    def reset(self):
        """
        Reset the environment and return a dummy observation.
        """
        self._create_initial_board()
        self.current_player: int = 0
        self.done: bool = False
        self.info: dict = {}
        # load deck
        self.deck = load_deck(deck_config_path=CONFIG['deck'])
        # deal hands
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

    def step(self, action: int):
        """
        Process an action (placeholder) and update the game state.
        """
        reward = 0
        self.current_player = (self.current_player + 1) % self.num_players
        # play given card from player's hand
        played_card: Card = self.player_hands[self.current_player].pop(action)
        return self._get_obs(), reward, self.done, False, self.info

    def render(self, mode: str = 'human') -> None:
        """
        Render the board state to the console for debugging.
        """
        print("Board:")
        for pos, card in self.board.items():
            print(f"Position {pos}: {card}")
        print(f"Current player: {self.current_player}")
        # print hands
        for player, hand in self.player_hands.items():
            print(f"Player {player} hand: {hand}")

    def is_adjacent_to_board(self, pos: tuple[int,int]) -> bool:
        """
        Check if pos is adjacent to at least one existing board position.
        """
        x, y = pos
        neighbors = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
        return any(n in self.board for n in neighbors)

    def place_card(self, card: Card, pos: tuple[int,int]) -> bool:
        """
        Attempt to place the given card at pos on the board.
        Return True if the placement is valid and performed, False otherwise.

        Placement rules:
         1) pos must be empty.
         2) pos must be adjacent to an existing board card.
         3) Edges must align with any existing neighbors.
         4) There must be a continuous path from the start tile to this newly placed card.
        """
        if pos in self.board:
            # Already occupied
            return False
        # Must be adjacent
        if not self.is_adjacent_to_board(pos):
            return False

        # Check edge alignment with each neighbor that's on the board.
        x, y = pos
        neighbors = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
        for npos in neighbors:
            if npos in self.board:
                neighbor_card = self.board[npos]
                if not edges_line_up(card, neighbor_card, pos, npos):
                    return False

        # Temporarily place the card
        card.x, card.y = pos
        self.board[pos] = card

        # Now check connectivity from the start tile to this new card.
        if not self.can_reach(pos):
            # revert placement
            del self.board[pos]
            return False

        # If we pass all checks, we keep the card in place.
        return True

    def can_reach(self, target_pos: tuple[int,int]) -> bool:
        """
        Check if there's a path from self.start_position to target_pos.
        We use BFS over the board. Two positions are connected if edges_line_up returns True.
        """
        if self.start_position not in self.board:
            return False
        if target_pos not in self.board:
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
            neighbors = [(cx-1,cy), (cx+1,cy), (cx,cy-1), (cx,cy+1)]
            for npos in neighbors:
                if npos not in visited and npos in self.board:
                    if edges_line_up(
                        self.board[current], self.board[npos], current, npos
                    ):
                        visited.add(npos)
                        queue.append(npos)
        return False
