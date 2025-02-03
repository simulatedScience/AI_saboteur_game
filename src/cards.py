# cards.py
"""
Module for defining card objects and random card generation functions for the Saboteur game.
"""

# Standard library imports
import random
import itertools

# Local imports
from .config import GUI_CONFIG
from .deck_config import load_deck_config

class Card:
    """
    Represents a Saboteur card.

    Attributes:
        type (str): 'start', 'path', 'dead-end', or 'goal'
        x (int | None): Board x-coordinate (if placed)
        y (int | None): Board y-coordinate (if placed)
        edges (dict[str, str]): Mapping for edges (keys: 'top', 'right', 'bottom', 'left').
            Each value is one of: 'path', 'dead-end', or 'wall'.
        connections (list[tuple[str, str]]): List of connections between edges.
        goal_type (str | None): For goal cards, either 'gold' or 'coal'.
        rotation (int): Either 0 or 180.
        hidden (bool): For goal cards, whether the card is face down.
        selected (bool): For in-hand cards, whether this card is selected.
    """
    def __init__(
            self,
            card_type: str,
            x: int | None = None,
            y: int | None = None,
            edges: dict[str, str] | None = None,
            connections: list[tuple[str, str]] | None = None,
            goal_type: str | None = None
        ) -> None:

        self.type: str = card_type
        self.x: int | None = x
        self.y: int | None = y
        self.edges: dict[str, str] = edges if edges is not None else get_random_edges()
        self.connections: list[tuple[str, str]] = connections if connections is not None else calculate_connections(self.edges)
        self.goal_type: str | None = goal_type
        self.rotation: int = 0  # 0 or 180
        self.hidden: bool = False  # For goal cards
        self.selected: bool = False  # For cards in hand
        # Optional: store the pixel position for cards in hand (set by the GUI)
        self.hand_pos: tuple[int, int] | None = None

        # Validate: if any edge is 'path', then there must be at least a second path.
        n_paths: int = sum(1 for v in self.edges.values() if v == 'path')
        if 0 < n_paths < 2:
            raise ValueError("Invalid card: if any edge is 'path', there must be at least two.")

    def __str__(self) -> str:
        return f"Card({self.type}) at ({self.x}, {self.y})"

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

    A valid configuration is one where either there are no 'path' edges,
    or at least two edges are 'path'.
    """
    while True:
        candidate_edges: dict[str, str] = {
            'top': random.choice(['path', 'wall', 'dead-end']),
            'right': random.choice(['path', 'wall', 'dead-end']),
            'bottom': random.choice(['path', 'wall', 'dead-end']),
            'left': random.choice(['path', 'wall', 'dead-end'])
        }
        n_paths: int = sum(1 for v in candidate_edges.values() if v == 'path')
        if n_paths == 0 or n_paths >= 2:
            return candidate_edges

def calculate_connections(edges: dict[str, str]) -> list[tuple[str, str]]:
    """
    Calculate connections for a card based on its edges.
    For every pair of edges that are 'path', assume they are connected.

    Args:
        edges (dict[str, str]): The edge configuration.

    Returns:
        list[tuple[str, str]]: List of connections (each as a tuple of edge names).
    """
    path_edges: list[str] = [edge for edge, etype in edges.items() if etype == 'path']
    connections: list[tuple[str, str]] = []
    # Connect each pair (unique combinations)
    for a, b in itertools.combinations(path_edges, 2):
        if a < b:
            connections.append((a, b))
        else:
            connections.append((b, a))
    if "dead-end" in edges.values():
        print("Connections for card:")
        for edge, edge_type in edges.items():
            print(f"{edge}: {edge_type}")
        print("Connections:")
        for conn in connections:
            print(conn)
    return connections

def load_deck(deck_config_path: str) -> list[Card]:
    """
    Load a given deck

    Args:
        deck_config (list[dict[str, str  |  int  |  dict]]): _description_

    Returns:
        list[Card]: _description_
    """
    deck_config = load_deck_config(deck_config_path)
    deck: list[Card] = []
    for card_type in deck_config:
        # extract card count
        count: int = card_type.pop('count')
        for _ in range(count):
            deck.append(
                Card(
                    card_type['type'],
                    edges=card_type['edges'],
                    connections=card_type.get('connections', None),
                    goal_type=None,
                )
            )
    random.shuffle(deck)
    return deck