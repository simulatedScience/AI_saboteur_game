
import random
from ..saboteur_env import SaboteurEnv
from ..cards import Card

class RuleBasedAgent:
    def __init__(self, env: SaboteurEnv) -> None:
        """
        Initialize the RuleBasedAgent.

        Args:
            env (SaboteurEnv): The game environment.
        """
        self.env = env

    def placement_penalty(self, card: Card, pos: tuple[int, int]) -> float:
        """
        Compute a penalty for placing a card at the given position if it would connect
        to an adjacent hidden goal via a dead-end edge.
        
        Args:
            card (Card): The card to be placed.
            pos (tuple[int, int]): The candidate board position for placement.
            
        Returns:
            float: A penalty value (0 if no penalty).
        """
        penalty = 0.0
        x, y = pos
        # For each adjacent cell, if it contains a hidden goal, check the touching edge.
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_pos = (x + dx, y + dy)
            if neighbor_pos in self.env.board:
                neighbor = self.env.board[neighbor_pos]
                if neighbor.type == "goal" and neighbor.hidden:
                    if dx == 1:
                        edge = card.edges["right"]
                    elif dx == -1:
                        edge = card.edges["left"]
                    elif dy == 1:
                        edge = card.edges["bottom"]
                    elif dy == -1:
                        edge = card.edges["top"]
                    else:
                        continue
                    if edge == "dead-end":
                        penalty += 100  # Arbitrary penalty value; adjust as needed.
        return penalty

    def act(self, player_index: int) -> tuple[int, tuple[int, int], int]:
        """
        Choose an action based on a heuristic that minimizes the maximum-norm
        distance from the placement position to the closest hidden goal, plus a penalty
        for using dead-end edges.

        Args:
            player_index (int): The index of the current player.

        Returns:
            tuple: (card_index, board position, orientation); if no valid move exists, returns a skip action.
        """
        hand = self.env.player_hands[player_index]
        best_action: tuple[int, tuple[int, int], int] | None = None
        best_value: float = float('inf')
        
        # Gather positions of all hidden goal cards.
        hidden_goals: list[tuple[int, int]] = [
            pos for pos, c in self.env.board.items() if c.type == "goal" and c.hidden
        ]
        
        for i, card in enumerate(hand):
            placements = self.env.get_valid_placements(card)
            for pos in placements:
                # Compute maximum-norm distance from pos to the closest hidden goal.
                if not hidden_goals:
                    distance = 0
                else:
                    distances = [max(abs(pos[0] - g[0]), abs(pos[1] - g[1])) for g in hidden_goals]
                    distance = min(distances)
                penalty = self.placement_penalty(card, pos)
                value = distance + penalty
                if value < best_value:
                    best_value = value
                    best_action = (i, pos, card.rotation)
        if best_action is not None:
            return best_action
        else:
            return (-1, (0, 0), 0)
