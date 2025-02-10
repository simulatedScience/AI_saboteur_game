# saboteur_env.py
"""
This module implements the Saboteur environment with efficient connectivity
updates and further optimizations. In particular, _is_valid_placement() now
leverages a precomputed mapping from candidate positions to connecting directions,
avoiding repeated iteration over all reachable edges.
    
Author: OpenAI-o3-mini
"""

import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import CONFIG, AI_CONFIG
from .cards import Card, load_deck, calculate_connections

# Global constants for directions.
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
    Saboteur environment implementing placement rules based on connectivity.
    
    A new card may only be placed at a position adjacent to an open “path” edge
    connected to the start. The new card must offer a connecting edge (a "path" or
    "dead‑end") facing that neighbor. For normal cards the touching edges must match:
      - walls must touch walls, and
      - non‑walls (path/dead‑end) must touch non‑walls.
    Hidden goal cards are ignored for placement restrictions; once uncovered, their
    connectivity is updated as usual.
    
    Attributes:
      num_players (int): Number of players.
      board (dict[tuple[int, int], Card]): Mapping of board positions to placed cards.
      reachable_edges (set[tuple[int, int, str]]): Open "path" edges (from uncovered cards)
          that are connected to the start.
      valid_positions (set[tuple[int, int]]): Candidate board positions (empty or containing a
          hidden goal) adjacent to a reachable edge.
      valid_positions_map (dict[tuple[int, int], set[str]]): Mapping from candidate position to the
          set of connecting directions (from reachable edges).
      start_position (tuple[int, int]): Fixed starting tile position.
      deck (list[Card]): The deck of cards.
      player_hands (dict[int, list[Card]]): Cards in hand for each player.
      current_player (int): Index of the active player.
      done (bool): Whether the game is over.
      info (dict): Additional information (e.g. final rewards).
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
        # reachable_edges: (x, y, direction) tuples from uncovered cards (and the start) that have a "path" edge.
        self.reachable_edges: set[tuple[int, int, str]] = set()
        # valid_positions: set of candidate positions for new card placement.
        self.valid_positions: set[tuple[int, int]] = set()
        # valid_positions_map: mapping from candidate position to set of connecting directions.
        self.valid_positions_map: dict[tuple[int, int], set[str]] = {}
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
        Initialize connectivity structures (reachable_edges and valid_positions_map)
        based on the start tile.
        """
        self.reachable_edges.clear()
        self.valid_positions.clear()
        self.valid_positions_map.clear()
        start_card: Card = self.board[self.start_position]
        # Only "path" edges propagate connectivity.
        for d in ("top", "right", "bottom", "left"):
            if start_card.edges[d] == "path":
                self.reachable_edges.add((self.start_position[0], self.start_position[1], d))
        self._compute_valid_positions_map()  # <-- Updated function

    def _compute_valid_positions_map(self) -> None:
        """
        Recompute valid board positions based on reachable_edges and store a mapping
        from each candidate position to the set of connecting directions.
        
        A candidate is valid if:
          - It is empty, or
          - It is occupied by a hidden goal card.
        """
        vp_map: dict[tuple[int, int], set[str]] = {}
        for (x, y, d) in self.reachable_edges:
            dx, dy = DIRECTION_DELTAS[d]
            candidate: tuple[int, int] = (x + dx, y + dy)
            if candidate not in self.board or (
                candidate in self.board and self.board[candidate].type == "goal" and self.board[candidate].hidden
            ):
                vp_map.setdefault(candidate, set()).add(d)
        self.valid_positions_map = vp_map
        self.valid_positions = set(vp_map.keys())

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
                goal_edges = start_edges.copy()  # Gold goal: all "path"
                goal_type = "gold"
            else:
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
          1. pos is a candidate position (i.e. present in valid_positions_map).
          2. At least one connecting direction d (from valid_positions_map) is such that the new card's
             edge opposite to d is a connector ("path" or "dead‑end").
          3. For each neighbor in all four directions, if a card exists (unless it is a hidden goal),
             the touching edges match (walls with walls; non‑walls with non‑walls).
        
        This function has been optimized by using a precomputed valid_positions_map and by caching local variables.
        
        Args:
          card (Card): The card to be placed.
          pos (tuple[int, int]): The target board position.
        
        Returns:
          bool: True if placement is valid, False otherwise.
        """
        # Use the precomputed valid_positions_map for connectivity check.
        vp_map = self.valid_positions_map
        if pos not in vp_map:
            return False

        # Allowed connecting edge types.
        allowed = {"path", "dead-end"}
        c_edges = card.edges  # Cache the card's edges.
        # For any connecting direction d (from the neighbor that provided pos),
        # the new card's edge opposite to d must be in allowed.
        if not any(c_edges[OPPOSITE_EDGE[d]] in allowed for d in vp_map[pos]):
            return False

        # Now check neighboring cards for edge matching.
        board = self.board  # Cache board
        opp = OPPOSITE_EDGE  # Local alias
        for d, delta in DIRECTION_DELTAS.items():
            nx = pos[0] + delta[0]
            ny = pos[1] + delta[1]
            neighbor_pos = (nx, ny)
            if neighbor_pos in board:
                neighbor = board[neighbor_pos]
                # Hidden goal cards do not restrict placement.
                if neighbor.type == "goal" and neighbor.hidden:
                    continue
                new_edge = c_edges[d]
                neighbor_edge = neighbor.edges[opp[d]]
                if new_edge == "wall":
                    if neighbor_edge != "wall":
                        return False
                else:  # new_edge is "path" or "dead-end"
                    if neighbor_edge not in allowed:
                        return False
        return True

    def get_valid_placements(self, card: Card) -> list[tuple[int, int]]:
        """
        Compute all board positions where the given card (with its current orientation)
        can be legally placed.
        
        Args:
          card (Card): The card to test.
        
        Returns:
          list[tuple[int, int]]: Valid board positions.
        """
        valid: list[tuple[int, int]] = []
        # We now use the precomputed valid_positions_map keys.
        for pos in self.valid_positions:
            if self._is_valid_placement(card, pos):
                valid.append(pos)
        return valid

    def _update_reachable_edges_after_placement(self, pos: tuple[int, int], card: Card) -> None:
        """
        Update connectivity after placing a card at pos.
        
        For each direction from pos:
          - If the neighboring cell is occupied by a card that is not a hidden goal,
            remove the reachable edge on that neighbor.
          - If the neighbor is empty or occupied by a hidden goal (treated as empty) and the placed card
            has a "path" edge on that side, add that edge.
        
        Finally, recompute valid_positions_map.
        
        Args:
          pos (tuple[int, int]): The position of the newly placed card.
          card (Card): The placed card.
        """
        for d, delta in DIRECTION_DELTAS.items():
            neighbor_pos = (pos[0] + delta[0], pos[1] + delta[1])
            if neighbor_pos in self.board:
                if not (self.board[neighbor_pos].type == "goal" and self.board[neighbor_pos].hidden):
                    self.reachable_edges.discard((neighbor_pos[0], neighbor_pos[1], OPPOSITE_EDGE[d]))
                else:
                    if card.edges[d] == "path":
                        self.reachable_edges.add((pos[0], pos[1], d))
            else:
                if card.edges[d] == "path":
                    self.reachable_edges.add((pos[0], pos[1], d))
        self._compute_valid_positions_map()

    def _update_card_connectivity(self, pos: tuple[int, int]) -> None:
        """
        Update connectivity for the card at pos.
        
        For an uncovered card (including uncovered goal cards), if a given side is "path"
        and the neighbor cell is empty or contains a hidden goal, add that edge to connectivity.
        
        Args:
          pos (tuple[int, int]): Position of the card to update.
        """
        card = self.board[pos]
        if card.hidden:
            return  # Do not propagate connectivity from hidden cards.
        for d, delta in DIRECTION_DELTAS.items():
            neighbor_pos = (pos[0] + delta[0], pos[1] + delta[1])
            if neighbor_pos not in self.board or (
                neighbor_pos in self.board and self.board[neighbor_pos].type == "goal" and self.board[neighbor_pos].hidden
            ):
                if card.edges[d] == "path":
                    self.reachable_edges.add((pos[0], pos[1], d))
        self._compute_valid_positions_map()

    def _adjust_goal_card(self, goal_card: Card, connecting_direction: str) -> None:
        """
        Uncover a hidden goal card and rotate it so that its edge facing the connecting card becomes "path".
        Then update its connectivity.
        
        Args:
          goal_card (Card): The hidden goal card.
          connecting_direction (str): The direction on the new card that touches the goal.
                                      The goal card is rotated so that its OPPOSITE_EDGE[connecting_direction]
                                      becomes "path".
        """
        goal_card.hidden = False
        required_edge: str = OPPOSITE_EDGE[connecting_direction]
        if goal_card.edges[required_edge] != "path":
            goal_card.rotate()
        goal_card.connections = calculate_connections(goal_card.edges)
        self._update_card_connectivity((goal_card.x, goal_card.y))

    def place_card(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Attempt to place the given card at pos.
        
        If placement is valid, the card is added to the board, any adjacent hidden goal cards
        touched via a "path" edge are uncovered (and rotated if needed), and connectivity is updated.
        
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

        # For each side where the new card has a "path" edge, if the neighbor is a hidden goal card,
        # uncover and adjust that goal card.
        for d, delta in DIRECTION_DELTAS.items():
            if card.edges[d] == "path":
                neighbor_pos = (pos[0] + delta[0], pos[1] + delta[1])
                if neighbor_pos in self.board:
                    neighbor = self.board[neighbor_pos]
                    if neighbor.type == "goal" and neighbor.hidden:
                        self._adjust_goal_card(neighbor, d)

        self._update_reachable_edges_after_placement(pos, card)
        return True

    def gold_reached(self) -> bool:
        """
        Check if the gold goal has been uncovered.
        
        Returns:
          bool: True if the gold goal has been uncovered, False otherwise.
        """
        for goal_pos in ((8, 0), (8, 2), (8, -2)):
            if goal_pos in self.board:
                goal_card: Card = self.board[goal_pos]
                if goal_card.type == "goal" and goal_card.goal_type == "gold" and not goal_card.hidden:
                    return True
        return False

    def compute_final_rewards(self, gold_reached: bool = True) -> dict[tuple, int]:
        """
        Compute final rewards based on finishing order. If gold was not reached, everyone gets reward 0.
        If goal was reached, 
        
        The winning player receives AI_CONFIG["final_reward_winner"] points,
        and each subsequent player (in play order) receives one point less (minimum 1).
        
        Args:
            gold_reached (bool): Whether the gold goal was reached.
        
        Returns:
          dict: Mapping from player index to reward.
        """
        rewards: dict[tuple, int] = {}
        if gold_reached:
            winner_reward: int = AI_CONFIG["final_reward_winner"]
            for i in range(self.num_players):
                rewards[i] = max(1, winner_reward - i)
        else:
            for i in range(self.num_players):
                rewards[i] = 0
        return rewards

    def step(self, action: tuple[int, tuple[int, int], int]) -> tuple[np.ndarray, int, bool, bool, dict[str, any]]:
        """
        Process an action.
        
        The action is a tuple: (card_index, board position, orientation).
        A card_index of -1 indicates a skip action.
        Invalid actions are penalized with -20.
        After processing the action, the board is rendered for debugging.
        
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
                self.info["final_rewards"] = self.compute_final_rewards(gold_reached=False)
            return self._get_obs(), 0, self.done, False, self.info

        current_hand: list[Card] = self.player_hands[self.current_player]
        if card_index < 0 or card_index >= len(current_hand):
            print(f"Invalid card index: {card_index}")
            return self._get_obs(), -20, self.done, False, self.info

        played_card: Card = current_hand[card_index]
        while played_card.rotation != orientation:
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

        # Game end conditions: gold goal uncovered or all hands empty.
        gold_reached: bool = self.gold_reached()
        all_hands_empty: bool = all(len(hand) == 0 for hand in self.player_hands.values())
        if all_hands_empty or gold_reached:
            self.done = True
            self.info["final_rewards"] = self.compute_final_rewards(gold_reached)
        else:
            self.done = False
            self.current_player = (self.current_player + 1) % self.num_players
        # if self.done and gold_reached:
        #     self.render()
        # render board when gold was reached
        return self._get_obs(), 0, self.done, False, self.info

    def _get_obs(self) -> np.ndarray:
        """
        Return a dummy observation.
        
        Returns:
          np.ndarray: Dummy observation.
        """
        return np.array([0])

    def render(self, separate_cards: bool = False) -> None:
        """
        Render the board state to the console for debugging.
        
        Args:
          separate_cards (bool): If True, render space between cards.
        """
        board_render = {}
        for pos, card in self.board.items():
            board_render[pos] = self.render_card(card)

        # Determine board dimensions.
        if self.board:
            min_x = min(pos[0] for pos in self.board.keys())
            max_x = max(pos[0] for pos in self.board.keys())
            min_y = min(pos[1] for pos in self.board.keys())
            max_y = max(pos[1] for pos in self.board.keys())
        else:
            min_x = max_x = min_y = max_y = 0

        # Render row by row.
        for y in range(min_y, max_y + 1):
            row_top = ""
            row_middle = ""
            row_bottom = ""
            for x in range(min_x, max_x + 1):
                if (x, y) in board_render:
                    card_render = board_render[(x, y)]
                    row_top += card_render[0] + (" " if separate_cards else "")
                    row_middle += card_render[1] + (" " if separate_cards else "")
                    row_bottom += card_render[2] + (" " if separate_cards else "")
                else:
                    # Render empty cell with reachable edges.
                    edge_symbols: dict[str, str] = {}
                    for dir, delta in DIRECTION_DELTAS.items():
                        # Check if the neighbor cell (in reverse direction) has a reachable edge.
                        edge = "r" if ((x + delta[0], y + delta[1], OPPOSITE_EDGE[dir]) in self.reachable_edges) else " "
                        edge_symbols[dir] = edge
                    row_top += f" {edge_symbols['top']} " + (" " if separate_cards else "")
                    row_middle += f"{edge_symbols['left']} {edge_symbols['right']}" + (" " if separate_cards else "")
                    row_bottom += f" {edge_symbols['bottom']} " + (" " if separate_cards else "")
            print(row_top)
            print(row_middle)
            print(row_bottom)
            if separate_cards:
                print()  # Blank line between rows.
        print("-" * 50)

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

        # Use '#' in the center if connections exist, or blank otherwise.
        middle = " " if card.connections else "#"
        if card.type == "goal":
            if card.goal_type == "coal":
                middle = "C"
            elif card.goal_type == "gold":
                middle = "G"

        top = "#" + ("#" if card.edges["top"] == "wall" else " ") + "#"
        middle_row = ("#" if card.edges["left"] == "wall" else " ") + middle + ("#" if card.edges["right"] == "wall" else " ")
        bottom = "#" + ("#" if card.edges["bottom"] == "wall" else " ") + "#"
        return [top, middle_row, bottom]


if __name__ == "__main__":
    env = SaboteurEnv()
    env.reset()
    env.render(separate_cards=True)
