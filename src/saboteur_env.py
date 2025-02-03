# saboteur_env.py
# Standard library imports
import random
from collections import deque
from typing import Any

# Third-party imports
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Local imports
from .config import CONFIG
from .cards import Card, load_deck


def connected_via_allowed(card_a: Card, card_b: Card, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> bool:
    """
    Return True if the touching edges between card_a and card_b (at positions pos_a and pos_b)
    form an allowed connection for path connectivity.
    Allowed pairs for connectivity are:
      (path, path), (dead-end, path), or (path, dead-end).
    (Note: wall<->wall is allowed for placement but does not contribute to connectivity.)
    """
    ax, ay = pos_a
    bx, by = pos_b
    if bx - ax == 1 and by - ay == 0:
        edge_a, edge_b = card_a.edges['right'], card_b.edges['left']
    elif bx - ax == -1 and by - ay == 0:
        edge_a, edge_b = card_a.edges['left'], card_b.edges['right']
    elif by - ay == 1 and bx - ax == 0:
        edge_a, edge_b = card_a.edges['bottom'], card_b.edges['top']
    elif by - ay == -1 and bx - ax == 0:
        edge_a, edge_b = card_a.edges['top'], card_b.edges['bottom']
    else:
        return False

    allowed = {("path", "path"), ("dead-end", "path"), ("path", "dead-end")}
    return (edge_a, edge_b) in allowed


class SaboteurEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players: int | None = None) -> None:
        """
        Initialize the Saboteur environment.
        """
        super().__init__()
        self.num_players: int = num_players if num_players is not None else CONFIG['num_players']
        self.board: dict[tuple[int, int], Card] = {}
        self.start_position: tuple[int, int] = (0, 0)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)
        self.done: bool = False
        self.info: dict[str, Any] = {}
        self.consecutive_skips: int = 0
        self.last_valid_player: int | None = None  # record last player to make a valid move
        self._create_initial_board()
        self.current_player: int = 0

        self.deck = load_deck(deck_config_path=CONFIG['deck'])
        self.player_hands: dict[int, list[Card]] = {
            player_idx: [self.deck.pop() for _ in range(CONFIG['hand_size'])]
            for player_idx in range(self.num_players)
        }

    def _create_initial_board(self) -> None:
        """
        Create an initial board with a start tile at (0, 0) and three goal cards.
        """
        self.board.clear()
        start_edges = {'top': 'path', 'right': 'path', 'bottom': 'path', 'left': 'path'}
        start_card = Card('start', x=0, y=0, edges=start_edges)
        self.board[(0, 0)] = start_card
        self.start_position = (0, 0)

        goal_positions = [(8, 0), (8, 2), (8, -2)]
        gold_index = random.randint(0, len(goal_positions) - 1)
        for idx, pos in enumerate(goal_positions):
            coal_edges = {'top': 'wall', 'right': 'wall', 'bottom': 'path', 'left': 'path'}
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
        Reset the environment.
        """
        self._create_initial_board()
        self.current_player = 0
        self.done = False
        self.info = {}
        self.consecutive_skips = 0
        self.last_valid_player = None
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
        Simulate connectivity from the start tile to target_pos using allowed connectivity.
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
                    if connected_via_allowed(self.board[current], self.board[npos], current, npos):
                        visited.add(npos)
                        queue.append(npos)
        return False

    def can_reach(self, target_pos: tuple[int, int]) -> bool:
        """
        Check connectivity from the start tile to target_pos.
        """
        return self._simulated_can_reach(target_pos)

    def _is_valid_placement(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Test if placing card at pos is valid.
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
                if npos[0] - x == 1:
                    my_edge, neighbor_edge = card.edges['right'], neighbor_card.edges['left']
                elif npos[0] - x == -1:
                    my_edge, neighbor_edge = card.edges['left'], neighbor_card.edges['right']
                elif npos[1] - y == 1:
                    my_edge, neighbor_edge = card.edges['bottom'], neighbor_card.edges['top']
                elif npos[1] - y == -1:
                    my_edge, neighbor_edge = card.edges['top'], neighbor_card.edges['bottom']
                else:
                    continue
                # For non-goal neighbors, enforce allowed pairs (using our edges_match-like logic)
                if neighbor_card.type != "goal":
                    # Allowed pairs for placement (for non-goal): (path, path), (path, dead-end), (dead-end, path), (wall, wall)
                    allowed = {("path", "path"), ("path", "dead-end"), ("dead-end", "path"), ("wall", "wall")}
                    if (my_edge, neighbor_edge) not in allowed:
                        return False

        # Temporarily add card to test connectivity.
        self.board[pos] = card
        reachable = self._simulated_can_reach(pos)
        del self.board[pos]
        return reachable

    def get_valid_placements(self, card: Card) -> list[tuple[int, int]]:
        """
        Return all board positions where the given card (with its current orientation) can be placed.
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
        Check if pos is adjacent to any placed card.
        """
        x, y = pos
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return any(n in self.board for n in neighbors)

    def place_card(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Place the card at pos if valid. Also, uncover any adjacent goal card if connected via a path.
        """
        if not self._is_valid_placement(card, pos):
            return False

        card.x, card.y = pos
        self.board[pos] = card

        x, y = pos
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for npos in neighbors:
            if npos in self.board:
                neighbor_card = self.board[npos]
                if neighbor_card.type == "goal" and neighbor_card.hidden:
                    if npos[0] - x == 1:
                        my_edge, neighbor_edge = card.edges['right'], neighbor_card.edges['left']
                    elif npos[0] - x == -1:
                        my_edge, neighbor_edge = card.edges['left'], neighbor_card.edges['right']
                    elif npos[1] - y == 1:
                        my_edge, neighbor_edge = card.edges['bottom'], neighbor_card.edges['top']
                    elif npos[1] - y == -1:
                        my_edge, neighbor_edge = card.edges['top'], neighbor_card.edges['bottom']
                    else:
                        continue
                    # If the new card’s edge is a path, uncover the goal.
                    if my_edge == "path":
                        if neighbor_card.goal_type == "gold":
                            neighbor_card.hidden = False
                        elif neighbor_card.goal_type == "coal":
                            # For coal, if the edge pair is not allowed, rotate the goal card once.
                            allowed = {("path", "path"), ("path", "dead-end"), ("dead-end", "path"), ("wall", "wall")}
                            if (my_edge, neighbor_edge) not in allowed:
                                neighbor_card.rotate()
                            neighbor_card.hidden = False
        return True

    def compute_final_rewards(self) -> dict[int, int]:
        """
        Compute final rewards based on the finishing order.
        The winning player (the last valid mover) gets num_players points; then, in turn order,
        each subsequent player gets one point less.
        """
        winning_player = self.last_valid_player if self.last_valid_player is not None else self.current_player
        rewards = {}
        for i in range(self.num_players):
            player = (winning_player + i) % self.num_players
            rewards[player] = self.num_players - i
        return rewards

    def step(self, action: tuple[int, tuple[int, int], int]) -> tuple[np.ndarray, int, bool, bool, dict]:
        """
        Process an action.
          - If action[0] (card_index) is -1, it is a skip turn.
          - Otherwise, attempt to place the selected card.
        Invalid actions are punished with -20.
        On a valid move, if the game ends, final rewards are computed.
        """
        card_index, pos, orientation = action

        # Skip action
        if card_index == -1:
            self.consecutive_skips += 1
            self.current_player = (self.current_player + 1) % self.num_players
            if self.consecutive_skips >= self.num_players:
                self.done = True
                self.info["final_rewards"] = self.compute_final_rewards()
            return self._get_obs(), 0, self.done, False, self.info

        current_hand: list[Card] = self.player_hands[self.current_player]
        if card_index < 0 or card_index >= len(current_hand):
            print("Invalid card index!")
            return self._get_obs(), -20, self.done, False, self.info

        played_card: Card = current_hand[card_index]
        if played_card.rotation != orientation:
            played_card.rotate()

        success: bool = self.place_card(played_card, pos)
        if not success:
            print("Invalid card placement!")
            return self._get_obs(), -20, self.done, False, self.info

        # Valid move: reset consecutive skip counter.
        self.consecutive_skips = 0
        self.last_valid_player = self.current_player

        # Remove the played card and draw a new one if possible.
        del current_hand[card_index]
        if self.deck:
            new_card: Card = self.deck.pop()
            current_hand.append(new_card)

        # Check termination conditions.
        all_hands_empty = all(len(hand) == 0 for hand in self.player_hands.values())
        gold_reached = False
        for board_pos, card in self.board.items():
            if card.type == 'goal' and card.goal_type == 'gold' and not card.hidden:
                if self.can_reach(board_pos):
                    gold_reached = True
                    break

        if all_hands_empty or gold_reached:
            self.done = True
            self.info["final_rewards"] = self.compute_final_rewards()
        else:
            self.done = False
            self.current_player = (self.current_player + 1) % self.num_players

        return self._get_obs(), 0, self.done, False, self.info

    def render(self, mode: str = 'human') -> None:
        """
        Render board state to the console.
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