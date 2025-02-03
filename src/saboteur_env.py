# saboteur_env.py
"""
This module implements the Saboteur environment with advanced connectivity checking,
skip actions, and reward assignment. Connectivity is determined using each card's
internal connections property. This module uses Python 3.11 built‑in typehints for
all function signatures.

Author: OpenAI-o3-mini
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


def get_opposite_edge(edge: str) -> str:
    """
    Return the opposite edge for a given edge.

    Args:
        edge (str): One of "top", "right", "bottom", "left".

    Returns:
        str: The opposite edge.
    """
    opposites: dict[str, str] = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    return opposites.get(edge, "")


class SaboteurEnv(gym.Env):
    """
    Saboteur environment implementing placement rules, connectivity checking using
    card connections, skip actions, and reward assignment.

    Attributes:
        num_players (int): Number of players.
        board (dict[tuple[int, int], Card]): Mapping of board positions to placed cards.
        start_position (tuple[int, int]): Fixed starting tile position.
        deck (list[Card]): The deck of cards.
        player_hands (dict[int, list[Card]]): Cards in hand for each player.
        current_player (int): Index of the active player.
        done (bool): Whether the game is over.
        info (dict): Additional information (e.g. final rewards).
        consecutive_skips (int): Count of consecutive skip actions.
        last_valid_player (int | None): Last player to make a valid move.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_players: int | None = None) -> None:
        """
        Initialize the Saboteur environment.

        Args:
            num_players (int | None): Number of players (defaults from CONFIG).
        """
        super().__init__()
        self.num_players: int = num_players if num_players is not None else CONFIG["num_players"]
        self.board: dict[tuple[int, int], Card] = {}
        self.start_position: tuple[int, int] = (0, 0)
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)
        self.done: bool = False
        self.info: dict[str, any] = {}
        self.consecutive_skips: int = 0
        self.last_valid_player: int | None = None

        self._create_initial_board()
        self.current_player: int = 0

        self.deck: list[Card] = load_deck(deck_config_path=CONFIG["deck"])
        self.player_hands: dict[int, list[Card]] = {
            player_idx: [self.deck.pop() for _ in range(CONFIG["hand_size"])]
            for player_idx in range(self.num_players)
        }

    def _create_initial_board(self) -> None:
        """
        Create the initial board with a start tile at (0, 0) and three goal cards.
        The start tile is a four-way tile (all edges "path") and the goal cards start hidden.
        """
        self.board.clear()
        start_edges: dict[str, str] = {"top": "path", "right": "path", "bottom": "path", "left": "path"}
        start_card: Card = Card("start", x=0, y=0, edges=start_edges)
        self.board[(0, 0)] = start_card
        self.start_position = (0, 0)

        goal_positions: list[tuple[int, int]] = [(8, 0), (8, 2), (8, -2)]
        gold_index: int = random.randint(0, len(goal_positions) - 1)
        for idx, pos in enumerate(goal_positions):
            coal_edges: dict[str, str] = {"top": "wall", "right": "wall", "bottom": "path", "left": "path"}
            goal_card: Card = Card(
                "goal",
                x=pos[0],
                y=pos[1],
                edges=start_edges if idx == gold_index else coal_edges,
                connections=[],
                goal_type="gold" if idx == gold_index else "coal"
            )
            goal_card.hidden = True
            self.board[pos] = goal_card

    def reset(self) -> tuple[np.ndarray, dict[str, any]]:
        """
        Reset the environment to the initial state.

        Returns:
            tuple: A dummy observation and an empty info dict.
        """
        self._create_initial_board()
        self.current_player = 0
        self.done = False
        self.info = {}
        self.consecutive_skips = 0
        self.last_valid_player = None
        self.deck = load_deck(deck_config_path=CONFIG["deck"])
        self.player_hands = {
            player_idx: [self.deck.pop() for _ in range(CONFIG["hand_size"])]
            for player_idx in range(self.num_players)
        }
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        Return a dummy observation.

        Returns:
            np.ndarray: Dummy observation.
        """
        return np.array([0])

    def _propagate_internal(self, card: Card, initial_edges: set[str]) -> set[str]:
        """
        Given a card and an initial set of reachable edges, propagate internal connectivity
        using the card's connections property.

        Args:
            card (Card): The card whose internal connections to use.
            initial_edges (set[str]): The initial set of reachable edges.

        Returns:
            set[str]: The full set of reachable edges on the card.
        """
        reachable: set[str] = set(initial_edges)
        changed: bool = True
        while changed:
            changed = False
            for conn in card.connections:
                # Each connection is a sorted tuple (e1, e2) with e1 < e2.
                e1, e2 = conn
                if e1 in reachable and e2 not in reachable:
                    reachable.add(e2)
                    changed = True
                if e2 in reachable and e1 not in reachable:
                    reachable.add(e1)
                    changed = True
        return reachable

    def compute_reachable_edges(self) -> dict[tuple[int, int], set[str]]:
        """
        Compute the reachable edges for each card on the board starting from the start tile.
        Connectivity is propagated using each card's internal connections.

        Returns:
            dict[tuple[int, int], set[str]]: Mapping from board positions to the set of edges
                that are connected to the start tile.
        """
        reachable_edges: dict[tuple[int, int], set[str]] = {}
        for pos in self.board.keys():
            reachable_edges[pos] = set()

        # Initialize the start tile with all non-wall edges.
        start_card: Card = self.board[self.start_position]
        initial: set[str] = {edge for edge in ("top", "right", "bottom", "left") if start_card.edges[edge] != "wall"}
        reachable_edges[self.start_position] = self._propagate_internal(start_card, initial)

        # Propagate connectivity via BFS.
        queue: deque[tuple[int, int]] = deque([self.start_position])
        visited: set[tuple[int, int]] = {self.start_position}

        while queue:
            pos: tuple[int, int] = queue.popleft()
            current_card: Card = self.board[pos]
            for edge in reachable_edges[pos]:
                # Determine adjacent cell based on current edge.
                delta: tuple[int, int]
                if edge == "top":
                    delta = (0, -1)
                elif edge == "bottom":
                    delta = (0, 1)
                elif edge == "left":
                    delta = (-1, 0)
                elif edge == "right":
                    delta = (1, 0)
                else:
                    continue

                neighbor_pos: tuple[int, int] = (pos[0] + delta[0], pos[1] + delta[1])
                if neighbor_pos not in self.board:
                    continue
                neighbor: Card = self.board[neighbor_pos]
                neighbor_edge: str = get_opposite_edge(edge)
                if neighbor.edges[neighbor_edge] == "wall":
                    continue
                initial_neighbor: set[str] = {neighbor_edge}
                new_reachable: set[str] = self._propagate_internal(neighbor, initial_neighbor)
                if not new_reachable.issubset(reachable_edges[neighbor_pos]):
                    reachable_edges[neighbor_pos].update(new_reachable)
                    if neighbor_pos not in visited:
                        queue.append(neighbor_pos)
                        visited.add(neighbor_pos)
        return reachable_edges

    def can_reach(self, target_pos: tuple[int, int]) -> bool:
        """
        Determine if the card at target_pos is connected to the start tile via internal card connections.

        Args:
            target_pos (tuple[int, int]): The board position to test.

        Returns:
            bool: True if there is a continuous connection from the start tile, False otherwise.
        """
        if target_pos not in self.board:
            return False
        reachable: dict[tuple[int, int], set[str]] = self.compute_reachable_edges()
        return bool(reachable.get(target_pos, set()))

    def _is_valid_placement(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Test whether placing the given card at pos is valid. Checks that:
          1. pos is empty and adjacent to an existing card.
          2. For each non-goal neighbor, the touching edges form an allowed pair.
          3. The new placement yields connectivity from the start tile.

        Args:
            card (Card): The card to be placed.
            pos (tuple[int, int]): The target board position.

        Returns:
            bool: True if placement is valid, False otherwise.
        """
        if pos in self.board:
            return False
        if not self.is_adjacent_to_board(pos):
            return False

        x, y = pos
        for npos in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if npos in self.board:
                neighbor: Card = self.board[npos]
                if npos[0] - x == 1:
                    my_edge, neighbor_edge = card.edges["right"], neighbor.edges["left"]
                elif npos[0] - x == -1:
                    my_edge, neighbor_edge = card.edges["left"], neighbor.edges["right"]
                elif npos[1] - y == 1:
                    my_edge, neighbor_edge = card.edges["bottom"], neighbor.edges["top"]
                elif npos[1] - y == -1:
                    my_edge, neighbor_edge = card.edges["top"], neighbor.edges["bottom"]
                else:
                    continue
                if neighbor.type != "goal":
                    allowed: set[tuple[str, str]] = {("path", "path"), ("path", "dead-end"), ("dead-end", "path"), ("wall", "wall")}
                    if (my_edge, neighbor_edge) not in allowed:
                        return False

        # Temporarily add the card to check connectivity.
        self.board[pos] = card
        is_connected: bool = self.can_reach(pos)
        del self.board[pos]
        return is_connected

    def get_valid_placements(self, card: Card) -> list[tuple[int, int]]:
        """
        Compute all board positions where the given card (with its current orientation)
        can be legally placed.

        Args:
            card (Card): The card to test.

        Returns:
            list[tuple[int, int]]: Valid board positions.
        """
        candidate_positions: set[tuple[int, int]] = set()
        for (x, y) in self.board.keys():
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                pos: tuple[int, int] = (x + dx, y + dy)
                if pos not in self.board:
                    candidate_positions.add(pos)
        return [pos for pos in candidate_positions if self._is_valid_placement(card, pos)]

    def is_adjacent_to_board(self, pos: tuple[int, int]) -> bool:
        """
        Check if the given position is adjacent to any card on the board.

        Args:
            pos (tuple[int, int]): The board position to check.

        Returns:
            bool: True if at least one neighbor exists, False otherwise.
        """
        x, y = pos
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in self.board:
                return True
        return False

    def place_card(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Attempt to place the given card at pos. If valid, adds the card to the board and
        uncovers any adjacent hidden goal card (gold or coal) that is connected via a "path" edge.
        For coal goals, we update the card's connections so that its predefined paths take effect.
        
        Args:
            card (Card): The card to be placed.
            pos (tuple[int, int]): The target board position.
        
        Returns:
            bool: True if placement is successful, False otherwise.
        """
        if not self._is_valid_placement(card, pos):
            return False

        card.x, card.y = pos
        self.board[pos] = card

        # Import calculate_connections here so that uncovered goals can have their connectivity updated.
        from .cards import calculate_connections

        x, y = pos
        for npos in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if npos in self.board:
                neighbor: Card = self.board[npos]
                if neighbor.type == "goal" and neighbor.hidden:
                    # Determine which edge of the new card touches the neighbor.
                    if npos[0] - x == 1:
                        my_edge = card.edges["right"]
                    elif npos[0] - x == -1:
                        my_edge = card.edges["left"]
                    elif npos[1] - y == 1:
                        my_edge = card.edges["bottom"]
                    elif npos[1] - y == -1:
                        my_edge = card.edges["top"]
                    else:
                        continue
                    # If the new card's edge is "path", then uncover the goal.
                    if my_edge == "path":
                        # For coal (or even gold) goals, update connections so that they are treated as a valid path.
                        # This ensures that subsequent valid placement checks use the correct connectivity.
                        neighbor.connections = calculate_connections(neighbor.edges)
                        neighbor.hidden = False
        return True



    def compute_final_rewards(self) -> dict[int, int]:
        """
        Compute final rewards based on finishing order. The last valid mover receives
        num_players points; each subsequent player (in play order) receives one point less.

        Returns:
            dict[int, int]: Mapping from player index to reward.
        """
        winning_player: int = self.last_valid_player if self.last_valid_player is not None else self.current_player
        rewards: dict[int, int] = {}
        for i in range(self.num_players):
            player: int = (winning_player + i) % self.num_players
            rewards[player] = self.num_players - i
        return rewards

    def step(self, action: tuple[int, tuple[int, int], int]) -> tuple[np.ndarray, int, bool, bool, dict[str, any]]:
        """
        Process an action.

        The action is a tuple: (card_index, board position, orientation).
        If card_index is -1, the player skips their turn.
        Invalid actions are penalized with -20.
        On a valid move, if the game ends, final rewards are computed.

        Args:
            action (tuple[int, tuple[int, int], int]): The action tuple.

        Returns:
            tuple: (observation, reward, done flag, truncated flag, info dict)
        """
        card_index, pos, orientation = action

        # Skip turn action.
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

        # Remove the played card and draw a new one if available.
        del current_hand[card_index]
        if self.deck:
            new_card: Card = self.deck.pop()
            current_hand.append(new_card)

        # Check termination conditions.
        all_hands_empty: bool = all(len(hand) == 0 for hand in self.player_hands.values())
        gold_reached: bool = False
        for board_pos, card in self.board.items():
            if card.type == "goal" and card.goal_type == "gold" and not card.hidden:
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

    def render(self, mode: str = "human") -> None:
        """
        Render the board state to the console for debugging.

        Args:
            mode (str, optional): The render mode. Defaults to "human".
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
