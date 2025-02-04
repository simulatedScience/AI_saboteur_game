# deck_config.py
"""
This file contains the configuration for the deck of cards used in the Saboteur game.
"""
import json

def load_deck_config(
        file_path: str,
    ) -> list[dict[str, str | int | dict]]:
    """
    Load a deck configuration from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        list[dict[str, str | int | dict]]: List of card configurations. including keys:
        - 'type': 'path' | 'goal' | 'start'
        - 'edges': dict[str, str]
        - 'connections': list[tuple[str, str]] | None
        - 'count': int
    """
    with open(file_path, 'r') as file:
        return json.load(file)
