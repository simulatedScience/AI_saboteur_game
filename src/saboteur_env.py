# saboteur_env.py
"""
Saboteur environment implementing placement rules based on connectivity,
with further optimizations and a new dense intermediate reward signal.

A new card may only be placed at a position adjacent to an open “path” edge
connected to the start. The new card must offer a connecting edge ("path" or "dead-end")
facing that neighbor. For normal cards, the touching edges must match:
  - walls touch walls, and
  - non‑walls (path/dead‑end) touch non‑walls.
Hidden goal cards are ignored for placement restrictions; once uncovered,
their connectivity is updated as usual.

Additionally, a dense reward signal is computed based on the Manhattan distance
from the current frontier (valid placements) to any remaining hidden goal card.
This value (clamped to a maximum of 8) is stored in self.distance_to_goal.
After each placement, if the distance decreases, an intermediate reward is given:
    reward = (old_distance - new_distance) * AI_CONFIG["dist_reward_scale"]

If AI_CONFIG["dist_reward_scale"] is 0, no intermediate reward is provided.
The final reward for reaching the goal remains unchanged.

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

MAX_BOARD_CARDS: int = 100
CARD_FEATURES: int = 22
BOARD_STATE_SIZE: int = MAX_BOARD_CARDS * CARD_FEATURES
HAND_SIZE: int = CONFIG["hand_size"]
HAND_STATE_SIZE: int = HAND_SIZE * CARD_FEATURES
STATE_SIZE: int = BOARD_STATE_SIZE + HAND_STATE_SIZE
COORD_LOW: float = -20.0 # minimum valid x-coordinate
COORD_HIGH: float = 20.0 # maximum valid x-coordinate
COORD_RES: int = 41 # 

DEBUG: bool = False

class SaboteurEnv(gym.Env):
    """
    Saboteur environment implementing placement rules based on connectivity,
    optimized for RL training. In addition to the usual game logic, a dense intermediate
    reward signal is provided based on the Manhattan distance from the frontier (valid positions)
    to any remaining hidden goal card.
    
    Attributes:
      num_players (int): Number of players.
      board (dict[tuple[int, int], Card]): Mapping from board positions to placed cards.
      reachable_edges (set[tuple[int, int, str]]): Open "path" edges (from uncovered cards)
          connected to the start.
      valid_positions (set[tuple[int, int]]): Candidate board positions (empty or with a hidden goal)
          adjacent to a reachable edge.
      valid_positions_map (dict[tuple[int, int], set[str]]): Mapping from candidate position to the
          set of connecting directions.
      distance_to_goal (int): Current minimum Manhattan distance (clamped to 8) from the frontier
          to any hidden goal card.
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
        super().__init__()
        self.num_players: int = num_players if num_players is not None else CONFIG["num_players"]
        # state variables
        self.board: dict[tuple[int, int], Card] = {}
        self.reachable_edges: set[tuple[int, int, str]] = set()
        self.valid_positions: set[tuple[int, int]] = set()
        self.valid_positions_map: dict[tuple[int, int], set[str]] = {}
        self.distance_to_goal: int = 8  # Initialize to maximum (8)
        self.start_position: tuple[int, int] = (0, 0)
        # AI STate & Action space
        self.observation_space = gym.spaces.Box(
            low=-1.,
            high=1.,
            shape=(STATE_SIZE,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete(
            [
                HAND_SIZE + 1, # hand_size
                COORD_RES, # discrete x-coordinates
                COORD_RES, # discrete y-coordinates
                2, # Orientation
            ]
        )

        self.done: bool = False
        self.info: dict[str, any] = {}
        self.consecutive_skips: int = 0
        self.last_valid_player: int | None = None

        self._create_initial_board()
        self._initialize_connectivity()
        self.current_player: int = 0

        self.deck: list[Card] = load_deck(deck_config_path=CONFIG["deck"])
        self.player_hands: dict[int, list[Card]] = {
            player: [self.deck.pop() for _ in range(HAND_SIZE)]
            for player in range(self.num_players)
        }

    def _initialize_connectivity(self) -> None:
        """
        Initialize connectivity structures (reachable_edges and valid_positions_map)
        based on the start tile, and update the current distance-to-goal.
        """
        self.reachable_edges.clear()
        self.valid_positions.clear()
        self.valid_positions_map.clear()
        start_card: Card = self.board[self.start_position]
        for d in ("top", "right", "bottom", "left"):
            if start_card.edges[d] == "path":
                self.reachable_edges.add((self.start_position[0], self.start_position[1], d))
        self._compute_valid_positions_map()
        self._update_distance_to_goal()  # Set initial distance

    def _compute_valid_positions_map(self) -> None:
        """
        Recompute valid board positions based on reachable_edges and store a mapping
        from each candidate position to the set of connecting directions.
        A candidate is valid if it is empty or occupied by a hidden goal card.
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

    def _update_distance_to_goal(self) -> None:
        """
        Update self.distance_to_goal based on the current valid positions and remaining hidden goal cards.
        Computes the minimum Manhattan distance from any candidate position in valid_positions
        to any hidden goal card. Clamps the value to a maximum of 8.
        If no hidden goals remain or no valid positions exist, sets distance to 0.
        """
        hidden_goals = [card for card in self.board.values() if card.type == "goal" and card.hidden]
        if not hidden_goals or not self.valid_positions:
            self.distance_to_goal = 0
            return

        min_dist = float('inf')
        for pos in self.valid_positions:
            for goal in hidden_goals:
                d = abs(pos[0] - goal.x) + abs(pos[1] - goal.y)
                if d < min_dist:
                    min_dist = d
        self.distance_to_goal = int(min(min_dist, 8))

    def _create_initial_board(self) -> None:
        """
        Create the initial board with a start tile at (0, 0) and three goal cards.
        The start tile is a four‑way tile (all edges "path") and the goal cards start hidden.
        """
        self.board.clear()
        start_edges: dict[str, str] = {"top": "path", "right": "path", "bottom": "path", "left": "path"}
        start_card: Card = Card("start", x=0, y=0, edges=start_edges)
        self.board[(0, 0)] = start_card
        self.start_position = (0, 0)

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
            player: [self.deck.pop() for _ in range(HAND_SIZE)]
            for player in range(self.num_players)
        }
        return self._get_obs(), {}

    def _is_valid_placement(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Check whether placing the given card at pos is valid.
        
        A placement is valid if:
          1. pos is a candidate position (present in valid_positions_map).
          2. At least one connecting direction d (from valid_positions_map) is such that the new card's
             edge opposite to d is a connector ("path" or "dead‑end").
          3. For each neighbor in all four directions (except hidden goals), the touching edges match.
        """
        vp_map = self.valid_positions_map
        if pos not in vp_map:
            return False

        allowed = {"path", "dead-end"}
        c_edges = card.edges  # Cache card's edges.
        if not any(c_edges[OPPOSITE_EDGE[d]] in allowed for d in vp_map[pos]):
            return False

        board = self.board  # Local alias.
        opp = OPPOSITE_EDGE
        for d, delta in DIRECTION_DELTAS.items():
            neighbor_pos = (pos[0] + delta[0], pos[1] + delta[1])
            if neighbor_pos in board:
                neighbor = board[neighbor_pos]
                if neighbor.type == "goal" and neighbor.hidden:
                    continue
                new_edge = c_edges[d]
                neighbor_edge = neighbor.edges[opp[d]]
                if new_edge == "wall":
                    if neighbor_edge != "wall":
                        return False
                else:
                    if neighbor_edge not in allowed:
                        return False
        return True

    def get_valid_placements(self, card: Card) -> list[tuple[int, int]]:
        """
        Compute all board positions where the given card (with its current orientation)
        can be legally placed.
        """
        valid = []
        for pos in self.valid_positions:
            if self._is_valid_placement(card, pos):
                valid.append(pos)
        return valid

    def _update_reachable_edges_after_placement(self, pos: tuple[int, int], card: Card) -> None:
        """
        Update connectivity after placing a card at pos, then recompute valid_positions_map.
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
        Update connectivity for the card at pos (if uncovered).
        """
        card = self.board[pos]
        if card.hidden:
            return
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
        Uncover a hidden goal card and rotate it so that its edge facing the connecting card becomes "path",
        then update its connectivity.
        """
        goal_card.hidden = False
        required_edge = OPPOSITE_EDGE[connecting_direction]
        if goal_card.edges[required_edge] != "path":
            goal_card.rotate()
        goal_card.connections = calculate_connections(goal_card.edges)
        self._update_card_connectivity((goal_card.x, goal_card.y))

    def place_card(self, card: Card, pos: tuple[int, int]) -> bool:
        """
        Attempt to place the given card at pos. If placement is valid, add it to the board,
        adjust any adjacent hidden goal cards, update connectivity, and update distance-to-goal.
        """
        if not self._is_valid_placement(card, pos):
            return False

        card.x, card.y = pos
        self.board[pos] = card

        # Uncover adjacent hidden goal cards if connected by a "path" edge.
        for d, delta in DIRECTION_DELTAS.items():
            if card.edges[d] == "path":
                neighbor_pos = (pos[0] + delta[0], pos[1] + delta[1])
                if neighbor_pos in self.board:
                    neighbor = self.board[neighbor_pos]
                    if neighbor.type == "goal" and neighbor.hidden:
                        self._adjust_goal_card(neighbor, d)

        self._update_reachable_edges_after_placement(pos, card)
        # Update distance-to-goal after connectivity changes.
        self._update_distance_to_goal()
        return True

    def gold_reached(self) -> bool:
        """
        Check if the gold goal has been uncovered.
        """
        for goal_pos in ((8, 0), (8, 2), (8, -2)):
            if goal_pos in self.board:
                goal_card = self.board[goal_pos]
                if goal_card.type == "goal" and goal_card.goal_type == "gold" and not goal_card.hidden:
                    return True
        return False

    def compute_final_rewards(self, gold_reached: bool = True) -> dict[int, int]:
        """
        Compute final rewards based on finishing order. If gold was not reached, everyone gets 0.
        Otherwise, the winning player receives AI_CONFIG["final_reward_winner"] points, and each subsequent
        player receives one point less (minimum 1).
        """
        rewards = {}
        if gold_reached:
            winner_reward = AI_CONFIG["final_reward_winner"]
            for i in range(self.num_players):
                rewards[i] = max(1, winner_reward - i)
        else:
            for i in range(self.num_players):
                rewards[i] = 0
        return rewards

    def _handle_skip_action(self) -> tuple[np.ndarray, float, bool, bool, dict[str, any]]:
        """
        Handle a skip action (card_index == -1).
        
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        self.consecutive_skips += 1
        self.current_player = (self.current_player + 1) % self.num_players
        reward: float = 0.0
        if self.consecutive_skips >= self.num_players:
            self.done = True
            reward = self.compute_final_rewards(gold_reached=False)[self.current_player]
            self.info["final_rewards"] = reward
        return self._get_obs(), reward, self.done, False, self.info

    def _handle_invalid_card_index(self, card_index: int, current_hand: list[Card]) -> tuple[np.ndarray, float, bool, bool, dict[str, any]]:
        """
        Handle an action with an invalid card index.
        
        Args:
            card_index (int): The invalid card index.
            current_hand (list[Card]): The current player's hand.
        
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        # print(f"Invalid card index: {card_index}")
        reward: float = -20.0
        self.current_player = (self.current_player + 1) % self.num_players
        return self._get_obs(), reward, self.done, False, self.info

    def _handle_invalid_placement(self, card_index: int, played_card: Card, old_distance: int, current_hand: list[Card]) -> tuple[np.ndarray, float, bool, bool, dict[str, any]]:
        """
        Handle an invalid card placement by substituting a valid placement if available.
        Applies a penalty for the invalid action and, if substitution succeeds, computes the intermediate reward.
        
        Args:
            card_index (int): Index of the card in hand.
            played_card (Card): The card that was attempted to be placed.
            old_distance (int): Distance-to-goal before placement.
            current_hand (list[Card]): The current player's hand.
        
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        # print(f"Invalid placement at {played_card.x, played_card.y}; substituting valid action.")
        invalid_penalty: float = -20.0
        valid_positions = self.get_valid_placements(played_card)
        if valid_positions:
            substitute_pos = valid_positions[0]
            sub_success: bool = self.place_card(played_card, substitute_pos)
            if sub_success:
                self.consecutive_skips = 0
                self.last_valid_player = self.current_player
                del current_hand[card_index]
                if self.deck:
                    new_card: Card = self.deck.pop()
                    current_hand.append(new_card)
                new_distance: int = self.distance_to_goal
                intermediate: float = (old_distance - new_distance) * AI_CONFIG["dist_reward_scale"] / max(self.distance_to_goal, 1)
                reward: float = invalid_penalty + intermediate
            else:
                reward = invalid_penalty
        else:
            reward = invalid_penalty
        self.current_player = (self.current_player + 1) % self.num_players
        return self._get_obs(), reward, self.done, False, self.info

    def _handle_valid_placement(self, card_index: int, played_card: Card, old_distance: int, current_hand: list[Card]) -> tuple[np.ndarray, float, bool, bool, dict[str, any]]:
        """
        Process a successful card placement.
        
        Removes the card from hand, draws a new one, computes the intermediate reward,
        and updates the current player.
        
        Args:
            card_index (int): The index of the placed card.
            played_card (Card): The card that was successfully placed.
            old_distance (int): Distance-to-goal before placement.
            current_hand (list[Card]): The current player's hand.
        
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        self.consecutive_skips = 0
        self.last_valid_player = self.current_player
        del current_hand[card_index]
        if self.deck:
            new_card: Card = self.deck.pop()
            current_hand.append(new_card)
        new_distance: int = self.distance_to_goal
        reward: float = (old_distance - new_distance) * AI_CONFIG["dist_reward_scale"] / max(self.distance_to_goal, 1)
        reached_gold: bool = self.gold_reached()
        all_hands_empty: bool = all(len(hand) == 0 for hand in self.player_hands.values())
        if all_hands_empty or reached_gold:
            self.done = True
            reward = self.compute_final_rewards(reached_gold)[self.current_player]
            self.info["final_rewards"] = reward
        else:
            self.done = False
            self.current_player = (self.current_player + 1) % self.num_players
        return self._get_obs(), reward, self.done, False, self.info

    def step(self,
            action: tuple[int, int, int, int],
            ) -> tuple[np.ndarray, float, bool, bool, dict[str, any]]:
        """
        Process an action.
        
        The action is now a tuple: (card_index, x_index, y_index, orientation).
            - If card_index == HAND_SIZE, then this is a skip action.
            - Otherwise, the card selected is current_hand[card_index].
            - The desired board position is computed as:
                (COORD_LOW + x_index, COORD_LOW + y_index)
            - Orientation is binary.
          
        If the placement is invalid, then with probability AI_CONFIG["mask_dropout_prob"],
        a valid placement (from get_valid_placements) is substituted; otherwise, the invalid move is penalized.
        
        Args:
            action (tuple[int, tuple[int, int], int]): (card_index, board position, orientation)
        
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        card_index, x_idx, y_idx, orientation = action

        # Handle skip action:
        if card_index == HAND_SIZE:
            return self._handle_skip_action()

        current_hand: list[Card] = self.player_hands[self.current_player]
        if card_index < 0 or card_index >= len(current_hand):
            # print(f"Invalid card index: {card_index}")
            return self._handle_invalid_card_index(card_index, current_hand)

        played_card: Card = current_hand[card_index]
        # Adjust card orientation if needed
        if played_card.rotation != orientation:
            played_card.rotate()

        # Compute desired placement from x_idx and y_idx (step size = 1)
        pos: tuple[int, int] = (COORD_LOW + x_idx, COORD_LOW + y_idx)
        old_distance: int = self.distance_to_goal

        success: bool = self.place_card(played_card, pos)
        if not success:
            # print(f"Invalid placement of card at {pos} (card: {played_card})")
            # With probability mask_dropout_prob, substitute a valid move.
            if self.get_valid_placements(played_card) and random.random() < AI_CONFIG["mask_dropout_prob"]:
                return self._handle_invalid_placement(card_index, played_card, old_distance, current_hand)
            else:
                # Otherwise, simply penalize the move.
                penalty: float = -20.0
                self.current_player = (self.current_player + 1) % self.num_players
                return self._get_obs(), penalty, self.done, False, self.info
        else:
            return self._handle_valid_placement(card_index, played_card, old_distance, current_hand)


    def _get_obs(self) -> np.ndarray:
        """
        Get the current observation for the active player.
        """
        board_state = self._encode_board()
        hand_state = self._encode_hand(self.current_player)
        return np.concatenate([board_state, hand_state])

    def _encode_board(self) -> np.ndarray:
        cards = list(self.board.values())
        cards.sort(key=lambda c: ((c.x if c.x is not None else 0), (c.y if c.y is not None else 0)))
        encodings = []
        for card in cards[:MAX_BOARD_CARDS]:
            encodings.append(self._encode_card(card))
        while len(encodings) < MAX_BOARD_CARDS:
            encodings.append(np.zeros(CARD_FEATURES, dtype=np.float32))
        return np.concatenate(encodings)
        
    def _encode_hand(self, player_index: int) -> np.ndarray:
        hand = self.player_hands[player_index]
        encodings = []
        for card in hand:
            encodings.append(self._encode_card(card))
        while len(encodings) < HAND_SIZE:
            encodings.append(np.zeros(CARD_FEATURES, dtype=np.float32))
        return np.concatenate(encodings)
        
    def _encode_card(self, card: Card) -> np.ndarray:
        x = float(card.x) if card.x is not None else 0.0
        y = float(card.y) if card.y is not None else 0.0
        pos = [x, y]
        edges = []
        for edge in ["top", "right", "bottom", "left"]:
            etype = card.edges.get(edge, "wall")
            if etype == "wall":
                edges.extend([1, 0, 0])
            else:
                edges.extend([0, 1, 0])
        possible_connections = [("left","right"), ("left","top"), ("left","bottom"),
                                ("right","top"), ("right","bottom"), ("top","bottom")]
        conn = [1 if pair in card.connections else 0 for pair in possible_connections]
        hidden_goal = 1 if (card.type == "goal" and card.hidden) else 0
        start_flag = 1 if card.type == "start" else 0
        flags = [hidden_goal, start_flag]
        return np.array(pos + edges + conn + flags, dtype=np.float32)

    def render(self, separate_cards: bool = False) -> None:
        """
        Render the board state to the console for debugging.
        """
        board_render = {}
        for pos, card in self.board.items():
            board_render[pos] = self.render_card(card)

        if self.board:
            min_x = min(pos[0] for pos in self.board.keys())
            max_x = max(pos[0] for pos in self.board.keys())
            min_y = min(pos[1] for pos in self.board.keys())
            max_y = max(pos[1] for pos in self.board.keys())
        else:
            min_x = max_x = min_y = max_y = 0

        for y in range(min_y, max_y + 1):
            row_top = ""
            row_middle = ""
            row_bottom = ""
            for x in range(min_x, max_x + 1):
                if (x, y) in board_render:
                    cr = board_render[(x, y)]
                    row_top += cr[0] + (" " if separate_cards else "")
                    row_middle += cr[1] + (" " if separate_cards else "")
                    row_bottom += cr[2] + (" " if separate_cards else "")
                else:
                    edge_symbols = {}
                    for d, delta in DIRECTION_DELTAS.items():
                        edge = "r" if ((x + delta[0], y + delta[1], OPPOSITE_EDGE[d]) in self.reachable_edges) else " "
                        edge_symbols[d] = edge
                    row_top += f" {edge_symbols['top']} " + (" " if separate_cards else "")
                    row_middle += f"{edge_symbols['left']} {edge_symbols['right']}" + (" " if separate_cards else "")
                    row_bottom += f" {edge_symbols['bottom']} " + (" " if separate_cards else "")
            print(row_top)
            print(row_middle)
            print(row_bottom)
            if separate_cards:
                print()
        print("-" * 50)

    def render_card(self, card: Card) -> list[str]:
        """
        Render a single card as a 3x3 square of characters.
        """
        if card.hidden:
            return ["###", "###", "###"]

        middle = " " if not card.connections else "#"
        if card.type == "goal":
            if card.goal_type == "coal":
                middle = "C"
            elif card.goal_type == "gold":
                middle = "G"

        top = "#" + ("#" if card.edges["top"] == "wall" else " ") + "#"
        mid = ("#" if card.edges["left"] == "wall" else " ") + middle + ("#" if card.edges["right"] == "wall" else " ")
        bottom = "#" + ("#" if card.edges["bottom"] == "wall" else " ") + "#"
        return [top, mid, bottom]


if __name__ == "__main__":
    env = SaboteurEnv()
    env.reset()
    env.render(separate_cards=False)
