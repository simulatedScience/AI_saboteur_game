# saboteur_env.py
"""
This module implements the Saboteur environment with an efficient connectivity
check using an internal set of reachable edges and pre‑computed valid placements.
Cards can only be placed if at least one “path” edge on the new card (the edge facing
the connecting neighbor) is connected to the start card. Walls and dead‑ends are not used
to propagate connectivity. Hidden goal cards are uncovered only when connected by a “path”
edge. Connectivity is updated efficiently after each placement.

Author: OpenAI-o3-mini
"""

import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import CONFIG, AI_CONFIG
from .cards import Card, load_deck, calculate_connections

# Global constants for directions.
# Convention: with (x,y) coordinates, y increases downward.
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

class SaboteurEnv(gym.Env):
    """
    Saboteur environment implementing card placement rules based on reachable edges.
    A new card can only be placed if at least one of its connecting edges (i.e. a "path" edge)
    touches a reachable edge from the start. Adjacent edges must match:
      - For normal cards: walls only with walls, and any non‑wall (path/dead‑end) only with non‑wall.
      - For a hidden goal, the new card must connect with a "path" edge.
    Connectivity is maintained via a set of reachable edges, and valid placement positions are pre‑computed.
    
    Attributes:
        num_players (int): Number of players.
        board (dict[tuple[int, int], Card]): Mapping from board positions to placed cards.
        reachable_edges (set[tuple[int, int, str]]): Set of open "path" edges (from cards connected
            to the start) represented as (x, y, direction).
        valid_positions (set[tuple[int, int]]): Set of empty positions adjacent to a reachable edge.
        start_position (tuple[int, int]): Fixed starting tile position.
        deck (list[Card]): The deck of cards.
        player_hands (dict[int, list[Card]]): Cards in hand for each player.
        current_player (int): Index of the active player.
        done (bool): Whether the game is over.
        info (dict): Additional information (e.g., final rewards).
        consecutive_skips (int): Count of consecutive skip actions.
        last_valid_player (int | None): The last player to make a valid move.
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
        # Set of tuples (x, y, direction) representing open "path" edges from cards connected to the start.
        self.reachable_edges: set[tuple[int, int, str]] = set()
        # Valid positions where a new card may be placed (computed from reachable_edges).
        self.valid_positions: set[tuple[int, int]] = set()
        self.start_position: tuple[int, int] = (0, 0)
        self.action_space = spaces.Discrete(10)  # Dummy; actions are handled externally.
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)
        self.done: bool = False
        self.info: dict[str, any] = {}
        self.consecutive_skips: int = 0
        self.last_valid_player: int | None = None
        
        self._create_initial_board()
        self._initialize_connectivity()
        self.current_player: int = 0
        
        self.deck: list[Card] = load_deck(deck_config_path=CONFIG["deck"])
        self.player_hands: dict[int, list[Card]] = {
            player: [self.deck.pop() for _ in range(CONFIG["hand_size"])]
            for player in range(self.num_players)
        }
    
    def _initialize_connectivity(self) -> None:
        """
        Initialize connectivity structures (reachable_edges and valid_positions)
        based on the start tile.
        """
        self.reachable_edges.clear()
        self.valid_positions.clear()
        start_card: Card = self.board[self.start_position]
        # Only "path" edges propagate connectivity.
        for d in ("top", "right", "bottom", "left"):
            if start_card.edges[d] == "path":
                self.reachable_edges.add((self.start_position[0], self.start_position[1], d))
        self._compute_valid_positions()
    
    def _compute_valid_positions(self) -> None:
        """
        Recompute valid board positions based on reachable_edges.
        For each reachable edge, the adjacent cell in that direction is a candidate.
        """
        self.valid_positions.clear()
        for (x, y, d) in self.reachable_edges:
            dx, dy = DIRECTION_DELTAS[d]
            candidate: tuple[int, int] = (x + dx, y + dy)
            if candidate not in self.board:
                self.valid_positions.add(candidate)
    
    def _create_initial_board(self) -> None:
        """
        Create the initial board with a start tile at (0, 0) and three goal cards.
        The start tile is a four‑way tile (all edges "path") and the goal cards start hidden.
        """
        self.board.clear()
        # Start card: all edges "path".
        start_edges: dict[str, str] = {"top": "path", "right": "path", "bottom": "path", "left": "path"}
        start_card: Card = Card("start", x=0, y=0, edges=start_edges)
        self.board[(0, 0)] = start_card
        self.start_position = (0, 0)
        
        # Create three goal cards at fixed positions.
        goal_positions: list[tuple[int, int]] = [(8, 0), (8, 2), (8, -2)]
        gold_index: int = random.randint(0, len(goal_positions) - 1)
        for idx, pos in enumerate(goal_positions):
            if idx == gold_index:
                # Gold goal: use "path" on all edges.
                goal_edges = start_edges.copy()
                goal_type = "gold"
            else:
                # Coal goal: example configuration.
                goal_edges = {"top": "wall", "right": "wall", "bottom": "path", "left": "path"}
                goal_type = "coal"
            goal_card: Card = Card("goal", x=pos[0], y=pos[1], edges=goal_edges, connections=[], goal_type=goal_type)
            goal_card.hidden = True
            self.board[pos] = goal_card
    
    def reset(self, seed: int = None, options: dict = {}) -> tuple[np.ndarray, dict[str, any]]:
        """
        Reset the environment to its initial state.
        
        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.
        
        Returns:
            tuple: (observation, info dictionary)
        """
        if options:
            for key, value in options.items():
                print(f"Setting option {key} to {value}")
        if seed is not None:
            random.seed(seed)
        self._create_initial_board()
        self._initialize_connectivity()
        self.current_player = 0
        self.done = False
        self.info = {}
        self.consecutive_skips = 0
        self.last_valid_player = None
        self.deck = load_deck(deck_config_path=CONFIG["deck"])
        self.player_hands = {
            player: [self.deck.pop() for _ in range(CONFIG["hand_size"])]
            for player in range(self.num_players)
        }
        return self._get_obs(), {}
    
    def _is_valid_placement(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Check whether placing the given card at pos is valid.
        
        A placement is valid if:
          1. pos is in the set of valid_positions (i.e. adjacent to a reachable "path" edge).
          2. The new card provides a connecting edge (i.e. "path" or "dead‑end") corresponding to that
             reachable edge.
          3. For each neighboring card, the touching edges match:
             - For normal cards: walls must touch walls and non‑walls (path/dead‑end) must touch non‑walls.
             - For a hidden goal: the new card’s touching edge must be "path" (i.e. a proper connection).
        
        Args:
            card (Card): The card to be placed.
            pos (tuple[int, int]): Target board position.
        
        Returns:
            bool: True if placement is valid, False otherwise.
        """
        if pos not in self.valid_positions:
            return False
        
        # Check connectivity using the reachable_edges.
        connected: bool = False
        for (rx, ry, d) in self.reachable_edges:
            dx, dy = DIRECTION_DELTAS[d]
            candidate = (rx + dx, ry + dy)
            if candidate == pos:
                # The new card’s connecting edge is the one opposite to d.
                # For connectivity, both "path" and "dead‑end" count.
                if card.edges[OPPOSITE_EDGE[d]] in {"path", "dead-end"}:
                    connected = True
                    break
        if not connected:
            return False
        
        # Check matching for each neighbor.
        for d, delta in DIRECTION_DELTAS.items():
            neighbor_pos: tuple[int, int] = (pos[0] + delta[0], pos[1] + delta[1])
            if neighbor_pos in self.board:
                neighbor: Card = self.board[neighbor_pos]
                # New card’s edge in direction d touches neighbor’s edge OPPOSITE_EDGE[d].
                new_edge: str = card.edges[d]
                neighbor_edge: str = neighbor.edges[OPPOSITE_EDGE[d]]
                if neighbor.type != "goal" or not neighbor.hidden:
                    if new_edge == "wall":
                        if neighbor_edge != "wall":
                            return False
                    else:
                        if neighbor_edge not in {"path", "dead-end"}:
                            return False
                else:
                    # For a hidden goal, require a proper connection: only "path" counts.
                    if new_edge != "path":
                        return False
        return True
    
    def get_valid_placements(self, card: Card) -> list[tuple[int, int]]:
        """
        Compute all board positions where the given card (with its current orientation)
        can be legally placed.
        
        Args:
            card (Card): The card to test.
        
        Returns:
            list[tuple[int, int]]: List of valid board positions.
        """
        valid: list[tuple[int, int]] = []
        for pos in self.valid_positions:
            if self._is_valid_placement(card, pos):
                valid.append(pos)
        return valid
    
    def _update_reachable_edges_after_placement(self, pos: tuple[int, int], card: Card) -> None:
        """
        Update the connectivity data structures after placing a card at pos.
        
        For each direction from pos:
          - If a neighboring cell already has a card, remove from that neighbor any reachable edge
            that pointed toward pos.
          - Otherwise, if the placed card has a "path" edge on that side, add it as a new reachable edge.
        Finally, recompute valid_positions.
        
        Args:
            pos (tuple[int, int]): The position of the newly placed card.
            card (Card): The placed card.
        """
        for d, delta in DIRECTION_DELTAS.items():
            neighbor_pos: tuple[int, int] = (pos[0] + delta[0], pos[1] + delta[1])
            if neighbor_pos in self.board:
                # Remove from the neighbor the reachable edge that would have pointed toward pos.
                self.reachable_edges.discard((neighbor_pos[0], neighbor_pos[1], OPPOSITE_EDGE[d]))
            else:
                # If no card occupies neighbor_pos and the placed card has a "path" edge on side d, add it.
                if card.edges[d] == "path":
                    self.reachable_edges.add((pos[0], pos[1], d))
        self._compute_valid_positions()
    
    def _adjust_goal_card(self, goal_card: Card, connecting_direction: str) -> None:
        """
        Uncover a hidden goal card and rotate it so that its edge facing the new card becomes "path".
        
        Args:
            goal_card (Card): The hidden goal card.
            connecting_direction (str): The direction on the new card that connects to the goal.
                                     The goal card should be rotated so that its edge
                                     OPPOSITE_EDGE[connecting_direction] is "path".
        """
        goal_card.hidden = False
        required_edge: str = OPPOSITE_EDGE[connecting_direction]
        # With only two orientations possible, one rotation is sufficient if needed.
        if goal_card.edges[required_edge] != "path":
            goal_card.rotate()
        goal_card.connections = calculate_connections(goal_card.edges)
    
    def place_card(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Attempt to place the given card at pos. If placement is valid, the card is added
        to the board, adjacent hidden goal cards (if connected via a "path" edge) are uncovered
        and rotated as needed, and connectivity is updated.
        
        Args:
            card (Card): The card to be placed.
            pos (tuple[int, int]): The target board position.
        
        Returns:
            bool: True if placement succeeds, False otherwise.
        """
        if not self._is_valid_placement(card, pos):
            return False
        
        card.x, card.y = pos
        self.board[pos] = card
        
        # Only check adjacent sides where the new card has a "path" edge.
        for d, delta in DIRECTION_DELTAS.items():
            if card.edges[d] == "path":
                neighbor_pos: tuple[int, int] = (pos[0] + delta[0], pos[1] + delta[1])
                if neighbor_pos in self.board:
                    neighbor: Card = self.board[neighbor_pos]
                    if neighbor.type == "goal" and neighbor.hidden:
                        # Uncover and rotate the goal so that its edge facing the new card becomes "path".
                        self._adjust_goal_card(neighbor, d)
        
        self._update_reachable_edges_after_placement(pos, card)
        return True
    
    def compute_final_rewards(self) -> dict[tuple, int]:
        """
        Compute final rewards based on finishing order.
        The winning player receives AI_CONFIG["final_reward_winner"] points,
        and each subsequent player (in play order) receives one point less (with a minimum of 1).
        
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
        A card_index of -1 indicates a skip action.
        Invalid actions are penalized with -20 points.
        On a valid move, if the game ends, final rewards are computed.
        
        Args:
            action (tuple[int, tuple[int, int], int]): The action tuple.
        
        Returns:
            tuple: (observation, reward, done flag, truncated flag, info dictionary)
        """
        card_index, pos, orientation = action
        
        if card_index == -1:
            self.consecutive_skips += 1
            self.current_player = (self.current_player + 1) % self.num_players
            if self.consecutive_skips >= self.num_players:
                self.done = True
                self.info["final_rewards"] = self.compute_final_rewards()
            return self._get_obs(), 0, self.done, False, self.info
        
        current_hand: list[Card] = self.player_hands[self.current_player]
        if card_index < 0 or card_index >= len(current_hand):
            print(f"Invalid card index: {card_index}")
            return self._get_obs(), -20, self.done, False, self.info
        
        played_card: Card = current_hand[card_index]
        # Rotate the card until its orientation matches the requested orientation.
        while played_card.rotation != orientation:
            played_card.rotate()
        
        success: bool = self.place_card(played_card, pos)
        if not success:
            print(f"Invalid card placement at x={pos[0]}, y={pos[1]}")
            return self._get_obs(), -20, self.done, False, self.info
        # DEBUG, TODO
        self.render()
        self.consecutive_skips = 0
        self.last_valid_player = self.current_player
        
        del current_hand[card_index]
        if self.deck:
            new_card: Card = self.deck.pop()
            current_hand.append(new_card)
        
        # Game end conditions: if the gold goal is uncovered or all hands are empty.
        gold_reached: bool = any(
            card.type == "goal" and card.goal_type == "gold" and not card.hidden
            for card in self.board.values()
        )
        all_hands_empty: bool = all(len(hand) == 0 for hand in self.player_hands.values())
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

    def render(self, separate_cards: bool =False) -> None:
        """
        Render the board state to the console for debugging.
        
        Args:
            separate_cards (bool): If True, render space between cards.
        """
        board_render = {}
        for pos, card in self.board.items():
            board_render[pos] = self.render_card(card)
        
        # Determine the board dimensions
        min_x = min(pos[0] for pos in self.board.keys())
        max_x = max(pos[0] for pos in self.board.keys())
        min_y = min(pos[1] for pos in self.board.keys())
        max_y = max(pos[1] for pos in self.board.keys())

        # Render the board row by row
        for y in range(min_y, max_y + 1):
            row_top = ""
            row_middle = ""
            row_bottom = ""
            for x in range(min_x, max_x + 1):
                if (x, y) in board_render:
                    card_render = board_render[(x, y)]
                    row_top += card_render[0] + " "*separate_cards
                    row_middle += card_render[1] + " "*separate_cards
                    row_bottom += card_render[2] + " "*separate_cards
                else: # render empty grid cell
                    # render reachable edges as "r"
                    # top_edge = "r" if (x, y, "top") in self.reachable_edges else " "
                    # right_edge = "r" if (x, y, "right") in self.reachable_edges else " "
                    # bottom_edge = "r" if (x, y, "bottom") in self.reachable_edges else " "
                    # left_edge = "r" if (x, y, "left") in self.reachable_edges else " "
                    edge_symbols: dict[str, str] = {}
                    for dir, delta in DIRECTION_DELTAS.items():
                        # delta = (-delta[0], -delta[1])  # reverse direction
                        edge = "r" if (x + delta[0], y + delta[1], OPPOSITE_EDGE[dir]) in self.reachable_edges else " "
                        # if edge == "r":
                        #     edge = "r"
                        edge_symbols[dir] = edge

                    # add the edges to the row strings
                    row_top += f" {edge_symbols['top']} " + " "*separate_cards
                    row_middle += f"{edge_symbols['left']} {edge_symbols['right']}" + " "*separate_cards
                    row_bottom += f" {edge_symbols['bottom']} " + " "*separate_cards
            print(row_top)
            print(row_middle)
            print(row_bottom)
            if separate_cards:
                print()  # Blank line between rows
        print("-"*50)

    def render_card(self, card: Card) -> list[str]:
        """
        Render a single card as a 3x3 square of characters.
        
        Args:
            card (Card): The card to render.
        
        Returns:
            list: A list of strings representing the 3x3 square.
        """
        if card.hidden:
            return ["###", "###", "###"]

        middle = " " if card.connections else "#"
        if card.type == "goal" and card.goal_type == "coal":
            middle = "C"
        elif card.type == "goal" and card.goal_type == "gold":
            middle = "G"
        
        top = "#" + ("#" if card.edges["top"] == "wall" else " ") + "#"
        middle_row = \
            ("#" if card.edges["left"] == "wall" else " ") \
            + middle \
            + ("#" if card.edges["right"] == "wall" else " ")
        bottom = "#" + ("#" if card.edges["bottom"] == "wall" else " ") + "#"
        return [top, middle_row, bottom]

if __name__ == "__main__":
    env = SaboteurEnv()
    env.reset()
    env.render()
