# saboteur_env.py
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


def edges_match(e1: str, e2: str) -> bool:
    """
    Returns True if the pair of edges is allowed.
    Allowed pairs: (path, path), (path, dead-end), (dead-end, path), (wall, wall).
    """
    if e1 == "wall" and e2 == "wall":
        return True
    if e1 == "path" and e2 in ("path", "dead-end"):
        return True
    if e1 == "dead-end" and e2 == "path":
        return True
    return False


def connected_via_path(card_a: Card, card_b: Card, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> bool:
    """
    Returns True if the connecting edges between card_a and card_b are both "path".
    This is used for connectivity (i.e. ensuring a continuous path from the start tile).
    """
    ax, ay = pos_a
    bx, by = pos_b
    if bx - ax == 1 and by - ay == 0:
        return card_a.edges['right'] == "path" and card_b.edges['left'] == "path"
    elif bx - ax == -1 and by - ay == 0:
        return card_a.edges['left'] == "path" and card_b.edges['right'] == "path"
    elif by - ay == 1 and bx - ax == 0:
        return card_a.edges['bottom'] == "path" and card_b.edges['top'] == "path"
    elif by - ay == -1 and bx - ax == 0:
        return card_a.edges['top'] == "path" and card_b.edges['bottom'] == "path"
    return False


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
        # The board maps (x, y) coordinates to Cards.
        self.board: dict[tuple[int, int], Card] = {}
        # The start tile is at (0, 0).
        self.start_position: tuple[int, int] = (0, 0)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)
        self.done: bool = False
        self.info: dict = {}
        self._create_initial_board()
        self.current_player: int = 0

        # Load deck and deal hands.
        self.deck = load_deck(deck_config_path=CONFIG['deck'])
        self.player_hands: dict[int, list[Card]] = {
            player_idx: [self.deck.pop() for _ in range(CONFIG['hand_size'])]
            for player_idx in range(self.num_players)
        }

    def _create_initial_board(self) -> None:
        """
        Create an initial board with a start tile at (0, 0) and three goal cards.
          - The start tile is a 4-way tile (all edges are "path").
          - Three goal cards are placed at fixed positions. One is gold (with all "path" edges),
            and the other two are coal (with preset coal edges). All goal cards start hidden.
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
            # For goal cards: gold gets the start_edges; coal gets preset coal_edges.
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

    def _simulated_can_reach(self, target_pos: tuple[int, int]) -> bool:
        """
        Simulate connectivity from the start tile to target_pos using only "path" edges.
        """
        if self.start_position not in self.board or target_pos not in self.board:
            return False

        queue = deque([self.start_position])
        visited = {self.start_position}

        while queue:
            current = queue.popleft()
            if current == target_pos:
                return True

            cx, cy = current
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                npos = (cx + dx, cy + dy)
                if npos in self.board and npos not in visited:
                    if connected_via_path(self.board[current], self.board[npos], current, npos):
                        visited.add(npos)
                        queue.append(npos)
        return False

    def can_reach(self, target_pos: tuple[int, int]) -> bool:
        """
        Check if there's a path from the start tile to target_pos.
        """
        return self._simulated_can_reach(target_pos)

    def _is_valid_placement(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Determine whether placing the given card at pos would be valid.
        This does not modify the permanent board.
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
                # Determine which edges touch.
                if npos[0] - x == 1:
                    my_edge = card.edges['right']
                    neighbor_edge = neighbor_card.edges['left']
                elif npos[0] - x == -1:
                    my_edge = card.edges['left']
                    neighbor_edge = neighbor_card.edges['right']
                elif npos[1] - y == 1:
                    my_edge = card.edges['bottom']
                    neighbor_edge = neighbor_card.edges['top']
                elif npos[1] - y == -1:
                    my_edge = card.edges['top']
                    neighbor_edge = neighbor_card.edges['bottom']
                else:
                    continue

                # For non-goal neighbors, enforce allowed edge pairs.
                if neighbor_card.type != "goal":
                    if not edges_match(my_edge, neighbor_edge):
                        return False
        # Temporarily add the card to test connectivity.
        self.board[pos] = card
        reachable = self._simulated_can_reach(pos)
        del self.board[pos]
        return reachable

    def get_valid_placements(self, card: Card) -> list[tuple[int, int]]:
        """
        Return a list of all board positions where the given card (with its current orientation)
        could be legally placed.
        """
        candidate_positions = set()
        for (x, y) in self.board.keys():
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                pos = (x + dx, y + dy)
                if pos not in self.board:
                    candidate_positions.add(pos)
        return [pos for pos in candidate_positions if self._is_valid_placement(card, pos)]

    def is_adjacent_to_board(self, pos: tuple[int, int]) -> bool:
        """
        Check if pos is adjacent to at least one existing board card.
        """
        x, y = pos
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return any(n in self.board for n in neighbors)

    def place_card(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Attempt to place the given card at pos on the board.
        Placement rules:
          1) pos must be empty and adjacent to an existing card.
          2) For each non-goal neighbor, the touching edges must form one of the allowed pairs.
          3) The placement must result in a continuous path (using only "path" edges)
             from the start tile to the new card.
          4) After placement, if a goal card is adjacent and the new card’s connecting edge is "path",
             then the goal is uncovered. In the case of a coal goal, if necessary the goal card is rotated.
        Returns True if placement is performed, False otherwise.
        """
        if not self._is_valid_placement(card, pos):
            return False

        # Place the card permanently.
        card.x, card.y = pos
        self.board[pos] = card

        x, y = pos
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for npos in neighbors:
            if npos in self.board:
                neighbor_card = self.board[npos]
                if neighbor_card.type == "goal" and neighbor_card.hidden:
                    # Determine the corresponding edges.
                    if npos[0] - x == 1:
                        my_edge = card.edges['right']
                        neighbor_edge = neighbor_card.edges['left']
                    elif npos[0] - x == -1:
                        my_edge = card.edges['left']
                        neighbor_edge = neighbor_card.edges['right']
                    elif npos[1] - y == 1:
                        my_edge = card.edges['bottom']
                        neighbor_edge = neighbor_card.edges['top']
                    elif npos[1] - y == -1:
                        my_edge = card.edges['top']
                        neighbor_edge = neighbor_card.edges['bottom']
                    else:
                        continue
                    if my_edge == "path":
                        if neighbor_card.goal_type == "gold":
                            neighbor_card.hidden = False
                        elif neighbor_card.goal_type == "coal":
                            # If the edge does not form an allowed pair, try rotating the coal card.
                            if not edges_match(my_edge, neighbor_edge):
                                neighbor_card.rotate()
                                # Recompute the neighbor_edge after rotation.
                                if npos[0] - x == 1:
                                    neighbor_edge = neighbor_card.edges['left']
                                elif npos[0] - x == -1:
                                    neighbor_edge = neighbor_card.edges['right']
                                elif npos[1] - y == 1:
                                    neighbor_edge = neighbor_card.edges['top']
                                elif npos[1] - y == -1:
                                    neighbor_edge = neighbor_card.edges['bottom']
                            neighbor_card.hidden = False
        return True

    def step(self, action: tuple[int, tuple[int, int], int]) -> tuple[np.ndarray, int, bool, bool, dict]:
        """
        Process an action consisting of:
          - card index in the current player's hand,
          - board position where the card is to be placed, and
          - desired card orientation (0 or 180).
        If placement is valid, the card is removed from the hand (and a new card drawn, if available),
        and the turn advances.
        Returns:
            observation, reward, done flag, truncated flag, info dict.
        """
        current_player: int = self.current_player
        current_hand: list[Card] = self.player_hands[current_player]
        card_index, pos, orientation = action

        if card_index < 0 or card_index >= len(current_hand):
            print("Invalid card index!")
            return self._get_obs(), -1, False, False, self.info

        played_card: Card = current_hand[card_index]
        if played_card.rotation != orientation:
            played_card.rotate()

        success: bool = self.place_card(played_card, pos)
        if not success:
            print("Invalid card placement!")
            return self._get_obs(), -1, False, False, self.info

        del current_hand[card_index]
        if self.deck:
            new_card: Card = self.deck.pop()
            current_hand.append(new_card)

        # Termination: if all hands are empty or if the gold card has been uncovered and reached.
        all_hands_empty: bool = all(len(hand) == 0 for hand in self.player_hands.values())
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


if __name__ == "__main__":
    env = SaboteurEnv()
    env.reset()
    env.render()