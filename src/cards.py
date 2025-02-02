# cards.py
"""
Module for defining card objects for the Saboteur game.
"""
import random
from typing import Optional

from .config import GUI_CONFIG

class Card:
    """
    Represents a Saboteur card.
    
    Attributes:
        type (str): 'start', 'path', 'dead-end', or 'goal'
        x (Optional[int]): Board x-coordinate (if placed)
        y (Optional[int]): Board y-coordinate (if placed)
        edges (dict[str, str]): A mapping for edges (keys: 'top', 'right', 'bottom', 'left').
                                 Each value is one of: 'path', 'dead-end', or 'wall'.
        connections (list[tuple[str, str]]): list of connections between edges.
        goal_type (Optional[str]): For goal cards: either 'gold' or 'coal'.
        rotation (int): Either 0 or 180.
        hidden (bool): For goal cards, whether the card is face down.
        selected (bool): For cards in-hand, whether this card is selected.
    """
    def __init__(
        self,
        card_type: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        edges: Optional[dict[str, str]] = None,
        connections: Optional[list[tuple[str, str]]] = None,
        goal_type: Optional[str] = None
    ) -> None:
        self.type: str = card_type
        self.x: Optional[int] = x
        self.y: Optional[int] = y
        self.edges: dict[str, str] = edges if edges is not None else get_random_edges()
        self.connections: list[tuple[str, str]] = connections if connections is not None else []
        self.goal_type: Optional[str] = goal_type
        self.rotation: int = 0  # 0 or 180
        self.hidden: bool = False  # For goal cards
        self.selected: bool = False  # For in-hand cards

        # Validate: if any edge is 'path', then there must be at least a second path.
        path_count: int = sum(1 for v in self.edges.values() if v == 'path')
        if 0 < path_count < 2:
            raise ValueError("Invalid card: if any edge is 'path', there must be at least two.")

    def __str__(self) -> str:
        return f"Card({self.type}) at ({self.x},{self.y})"

    def rotate(self) -> None:
        """
        Toggle rotation between 0 and 180°.
        This swaps the top and bottom edges and the left and right edges,
        and adjusts the connections accordingly.
        """
        self.rotation = 180 if self.rotation == 0 else 0
        new_edges: dict[str, str] = {
            'top': self.edges['bottom'],
            'right': self.edges['left'],
            'bottom': self.edges['top'],
            'left': self.edges['right']
        }
        self.edges = new_edges
        new_connections: list[tuple[str, str]] = []
        for conn in self.connections:
            a, b = conn
            a_new: str = self._rotated_edge(a)
            b_new: str = self._rotated_edge(b)
            new_connections.append((a_new, b_new))
        self.connections = new_connections

    def _rotated_edge(self, edge: str) -> str:
        """
        Helper function to return the new edge label after rotation.
        """
        if edge == 'top':
            return 'bottom'
        elif edge == 'bottom':
            return 'top'
        elif edge == 'left':
            return 'right'
        elif edge == 'right':
            return 'left'
        return edge


def get_random_edges() -> dict[str, str]:
    """
    Return a random, valid edge configuration for a card.
    """
    while True:
        candidate_edges = {
            'top': random.choice(['path', 'wall', 'dead-end']),
            'right': random.choice(['path', 'wall', 'dead-end']),
            'bottom': random.choice(['path', 'wall', 'dead-end']),
            'left': random.choice(['path', 'wall', 'dead-end'])
        }
        n_paths = sum(1 for v in candidate_edges.values() if v == 'path')
        if n_paths == 0 or n_paths >= 2:
            return candidate_edges
