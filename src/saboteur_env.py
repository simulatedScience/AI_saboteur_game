# saboteur_env.py
"""
This module implements the Saboteur environment with advanced connectivity checking,
skip actions, and reward assignment. Connectivity is now maintained using an
incremental graph that is updated whenever a new card is placed.
This module uses Python 3.11 built‑in typehints for all function signatures.

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
from .config import CONFIG, AI_CONFIG
from .cards import Card, load_deck, calculate_connections

# Global constants for directions and allowed pairs.
DIRECTION_DELTAS: dict[str, tuple[int, int]] = {
    "top": (0, -1),
    "bottom": (0, 1),
    "left": (-1, 0),
    "right": (1, 0)
}

OPPOSITE_EDGE: dict[str, str] = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left"
}

# For placement compatibility we require that touching edges be both "path" or
# (in our rules) we allow ("path", "dead-end") or ("dead-end", "path") as equivalent.
ALLOWED_PAIRS: set[tuple[str, str]] = {("path", "path"), ("path", "dead-end"), ("dead-end", "path"), ("wall", "wall")}


class SaboteurEnv(gym.Env):
    """
    Saboteur environment implementing placement rules, connectivity checking using
    an incremental graph, skip actions, and reward assignment.

    Attributes:
        num_players (int): Number of players.
        board (dict[tuple[int, int], Card]): Mapping of board positions to placed cards.
        graph (dict[tuple[int, int], set[tuple[int, int]]]): Connectivity graph over board positions.
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
        self.graph: dict[tuple[int, int], set[tuple[int, int]]] = {}  # incremental connectivity graph
        self.start_position: tuple[int, int] = (0, 0)
        self.action_space = spaces.Discrete(10)  # Dummy; actions are handled externally.
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
        # Initialize connectivity graph with the start tile.
        start_card: Card = self.board[self.start_position]
        self.graph = {self.start_position: set()}
        for edge in ("top", "right", "bottom", "left"):
            if start_card.edges[edge] != "wall":
                self.graph[self.start_position].add(edge)  # we store the edge label for debugging

    def _create_initial_board(self) -> None:
        """
        Create the initial board with a start tile at (0, 0) and three goal cards.
        The start tile is a four-way tile (all edges "path") and the goal cards start hidden.
        """
        self.board.clear()
        self.graph.clear()
        start_edges: dict[str, str] = {"top": "path", "right": "path", "bottom": "path", "left": "path"}
        start_card: Card = Card("start", x=0, y=0, edges=start_edges)
        self.board[(0, 0)] = start_card
        self.start_position = (0, 0)

        # Add the start position to the connectivity graph.
        self.graph[(0, 0)] = set()  # Initially empty; connectivity will be updated on placement

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

    def reset(self, seed: int = None, options: dict = {}) -> tuple[np.ndarray, dict[str, any]]:
        """
        Reset the environment to the initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.
        
        Returns:
            tuple: A dummy observation and an info dict.
        """
        if options:
            for key, value in options.items():
                print(f"Setting option {key} to {value}")
        if seed is not None:
            random.seed(seed)
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
        # Reinitialize connectivity graph for the start tile.
        start_card: Card = self.board[self.start_position]
        self.graph = {self.start_position: set()}
        for edge in ("top", "right", "bottom", "left"):
            if start_card.edges[edge] != "wall":
                self.graph[self.start_position].add(edge)
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """
        Return a dummy observation.

        Returns:
            np.ndarray: Dummy observation.
        """
        return np.array([0])

    def _is_valid_placement(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Test whether placing the given card at pos is valid. Checks that:
          1. pos is empty and adjacent to an existing card.
          2. For each non-goal neighbor, the touching edges form an allowed pair.
          3. The new placement, when simulated, is connected to the start tile.

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
        # Check edge compatibility with neighbors.
        for neighbor_pos in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor_pos in self.board:
                neighbor: Card = self.board[neighbor_pos]
                if neighbor_pos[0] - x == 1:
                    new_edge, neighbor_edge = card.edges["right"], neighbor.edges["left"]
                elif neighbor_pos[0] - x == -1:
                    new_edge, neighbor_edge = card.edges["left"], neighbor.edges["right"]
                elif neighbor_pos[1] - y == 1:
                    new_edge, neighbor_edge = card.edges["bottom"], neighbor.edges["top"]
                elif neighbor_pos[1] - y == -1:
                    new_edge, neighbor_edge = card.edges["top"], neighbor.edges["bottom"]
                else:
                    continue
                if neighbor.type != "goal":
                    if (new_edge, neighbor_edge) not in ALLOWED_PAIRS:
                        return False

        # Simulate placing the card: add it temporarily and update connectivity graph.
        self.board[pos] = card
        # Create a temporary graph copy for simulation.
        temp_graph = {node: neighbors.copy() for node, neighbors in self.graph.items()}
        self._simulate_update_connectivity(pos, card, temp_graph)
        is_connected = self._simulate_can_reach(pos, temp_graph)
        del self.board[pos]
        return is_connected

    def is_adjacent_to_board(self, pos: tuple[int, int]) -> bool:
        """
        Check if the given position is adjacent to any card on the board.

        Args:
            pos (tuple[int, int]): The board position.

        Returns:
            bool: True if at least one neighbor exists, False otherwise.
        """
        x, y = pos
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in self.board:
                return True
        return False

    def _simulate_update_connectivity(self, pos: tuple[int, int], card: Card, graph: dict[tuple[int, int], set[tuple[int, int]]]) -> None:
        """
        Simulate the connectivity graph update for a new card placed at pos.
        For each neighbor that exists in the board, if the card and neighbor connect,
        add a bidirectional edge in the temporary graph.
        
        Args:
            pos (tuple[int, int]): The position of the new card.
            card (Card): The new card.
            graph (dict): The temporary connectivity graph to update.
        """
        graph[pos] = set()
        x, y = pos
        for direction, delta in DIRECTION_DELTAS.items():
            neighbor_pos = (x + delta[0], y + delta[1])
            if neighbor_pos in self.board:
                neighbor = self.board[neighbor_pos]
                new_edge = card.edges[direction]
                neighbor_edge = neighbor.edges[OPPOSITE_EDGE[direction]]
                if (new_edge, neighbor_edge) in ALLOWED_PAIRS:
                    graph[pos].add(neighbor_pos)
                    if neighbor_pos in graph:
                        graph[neighbor_pos].add(pos)
                    else:
                        graph[neighbor_pos] = {pos}

    def _simulate_can_reach(self, target_pos: tuple[int, int], graph: dict[tuple[int, int], set[tuple[int, int]]]) -> bool:
        """
        Simulate a DFS on the temporary connectivity graph to check if target_pos is reachable from start_position.
        
        Args:
            target_pos (tuple[int, int]): The board position to test.
            graph (dict): The temporary connectivity graph.
        
        Returns:
            bool: True if target_pos is reachable, False otherwise.
        """
        if self.start_position not in graph:
            return False
        queue: deque[tuple[int, int]] = deque([self.start_position])
        visited: set[tuple[int, int]] = {self.start_position}
        while queue:
            current = queue.popleft()
            if current == target_pos:
                return True
            for neighbor in graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def _update_connectivity(self, pos: tuple[int, int], card: Card) -> None:
        """
        Update the global connectivity graph (self.graph) after placing a card at pos.
        This function checks all four neighbors and, if connectivity exists, adds bidirectional edges.
        
        Args:
            pos (tuple[int, int]): The position of the newly placed card.
            card (Card): The newly placed card.
        """
        self.graph[pos] = set()
        x, y = pos
        for direction, delta in DIRECTION_DELTAS.items():
            neighbor_pos = (x + delta[0], y + delta[1])
            if neighbor_pos in self.board:
                neighbor = self.board[neighbor_pos]
                new_edge = card.edges[direction]
                neighbor_edge = neighbor.edges[OPPOSITE_EDGE[direction]]
                if (new_edge, neighbor_edge) in ALLOWED_PAIRS:
                    self.graph[pos].add(neighbor_pos)
                    if neighbor_pos in self.graph:
                        self.graph[neighbor_pos].add(pos)
                    else:
                        self.graph[neighbor_pos] = {pos}

    def _update_reachable_after_placement(self, pos: tuple[int, int], card: Card) -> None:
        """
        Update the incremental connectivity graph after placing a card at pos.
        This method first updates the connectivity graph (self.graph) using _update_connectivity,
        then it does a DFS from the start tile to update a set self.reachable of reachable positions.
        
        Args:
            pos (tuple[int, int]): The position of the new card.
            card (Card): The new card.
        """
        self._update_connectivity(pos, card)
        # Recompute reachable nodes from the start using the graph.
        new_reachable: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque([self.start_position])
        while queue:
            current = queue.popleft()
            if current in new_reachable:
                continue
            new_reachable.add(current)
            for neighbor in self.graph.get(current, set()):
                if neighbor not in new_reachable:
                    queue.append(neighbor)
        # Save the reachable positions in self.reachable (as a dict mapping node -> dummy value)
        self.reachable = {node: None for node in new_reachable}

    def can_reach(self, target_pos: tuple[int, int]) -> bool:
        """
        Determine if the card at target_pos is connected to the start tile using the connectivity graph.
        
        Args:
            target_pos (tuple[int, int]): The board position to test.
        
        Returns:
            bool: True if target_pos is in self.reachable, False otherwise.
        """
        return target_pos in self.reachable

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
        valid_positions = [pos for pos in candidate_positions if self._is_valid_placement(card, pos)]
        return valid_positions

    def place_card(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Attempt to place the given card at pos. If valid, adds the card to the board and
        uncovers any adjacent hidden goal card (gold or coal) that is connected via a "path" edge.
        For coal goals, if the touching edge is not "path", rotate the coal card once so that it becomes "path",
        then update its connections accordingly.
        
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

        x, y = pos
        for npos in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if npos in self.board:
                neighbor: Card = self.board[npos]
                if neighbor.type == "goal" and neighbor.hidden:
                    if npos[0] - x == 1:
                        my_edge = card.edges["right"]
                        neighbor_edge = neighbor.edges["left"]
                    elif npos[0] - x == -1:
                        my_edge = card.edges["left"]
                        neighbor_edge = neighbor.edges["right"]
                    elif npos[1] - y == 1:
                        my_edge = card.edges["bottom"]
                        neighbor_edge = neighbor.edges["top"]
                    elif npos[1] - y == -1:
                        my_edge = card.edges["top"]
                        neighbor_edge = neighbor.edges["bottom"]
                    else:
                        continue
                    if my_edge == "path":
                        if neighbor.goal_type == "coal":
                            if neighbor_edge != "path":
                                neighbor.rotate()
                                if npos[0] - x == 1:
                                    neighbor_edge = neighbor.edges["left"]
                                elif npos[0] - x == -1:
                                    neighbor_edge = neighbor.edges["right"]
                                elif npos[1] - y == 1:
                                    neighbor_edge = neighbor.edges["top"]
                                elif npos[1] - y == -1:
                                    neighbor_edge = neighbor.edges["bottom"]
                        neighbor.connections = calculate_connections(neighbor.edges)
                        neighbor.hidden = False

        # Update the connectivity graph incrementally.
        self._update_reachable_after_placement(pos, card)
        return True

    def compute_final_rewards(self) -> dict[tuple, int]:
        """
        Compute final rewards based on finishing order.
        The winning player receives AI_CONFIG["final_reward_winner"] points,
        and each subsequent player (in play order) receives one point less,
        with a minimum of 1 point.
        
        Returns:
            dict: Mapping from player index to reward.
        """
        winner_reward: int = AI_CONFIG.get("final_reward_winner", self.num_players)
        rewards: dict[tuple, int] = {}
        winning_player: int = self.last_valid_player if self.last_valid_player is not None else self.current_player
        for i in range(self.num_players):
            player: int = (winning_player + i) % self.num_players
            rewards[player] = max(winner_reward - i, 1)
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

        if card_index == -1:
            self.consecutive_skips += 1
            self.current_player = (self.current_player + 1) % self.num_players
            if self.consecutive_skips >= self.num_players:
                self.done = True
                self.info["final_rewards"] = self.compute_final_rewards()
            return self._get_obs(), 0, self.done, False, self.info

        current_hand: list[Card] = self.player_hands[self.env_current_player()]
        if card_index < 0 or card_index >= len(current_hand):
            print(f"Invalid card index: {card_index}")
            return self._get_obs(), -20, self.done, False, self.info

        played_card: Card = current_hand[card_index]
        if played_card.rotation != orientation:
            played_card.rotate()

        success: bool = self.place_card(played_card, pos)
        if not success:
            print(f"Invalid card placement at x={pos[0]}, y={pos[1]}")
            return self._get_obs(), -20, self.done, False, self.info

        self.consecutive_skips = 0
        self.last_valid_player = self.current_player

        del current_hand[card_index]
        if self.deck:
            new_card: Card = self.deck.pop()
            current_hand.append(new_card)

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

    def _get_obs(self) -> np.ndarray:
        """
        Return a dummy observation.
        
        Returns:
            np.ndarray: Dummy observation.
        """
        return np.array([0])

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

def env_current_player(self) -> int:
    """
    Helper function to return the current player index.
    
    Returns:
        int: The current player.
    """
    return self.current_player

# Attach helper as a method.
SaboteurEnv.env_current_player = env_current_player

if __name__ == "__main__":
    env = SaboteurEnv()
    env.reset()
    env.render()
