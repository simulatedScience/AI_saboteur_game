# cards.py
from .config import config
import random

class Card:
    def __init__(self, card_type, x_index, y_index):
        """
        card_type: 'path', 'dead-end', 'start', 'gold', 'coal'
        x_index, y_index: grid indices for the board
        """
        self.type = card_type
        self.x_index = x_index
        self.y_index = y_index
        self.rotation = 0  # can be 0 or 180
        # For goal cards, we use a hidden flag to indicate if they are face-down.
        self.hidden = False  
        # For hand cards, we might later add additional properties.

    def __str__(self):
        return f"Card({self.type}) at ({self.x_index}, {self.y_index})"

def create_board():
    """
    Creates a board as a list of Card objects.
    For demonstration, cards alternate between 'path' and 'dead-end'.
    """
    board = []
    rows = config['boardRows']
    cols = config['boardCols']
    for row in range(rows):
        for col in range(cols):
            # Alternate card type as a simple demo
            card_type = 'path' if ((row + col) % 2 == 0) else 'dead-end'
            board.append(Card(card_type, col, row))
    return board

def create_game_board():
    """
    Creates the game board with a start card and 3 goal cards.
    The start card is placed at row 0, column = cols//2.
    The goal cards are placed at row (rows-1) at positions (mid-1, mid, mid+1)
    (if within bounds). One of the goal cards is randomly set to 'gold'; the others
    are set to 'coal'. Goal cards are initially hidden.
    """
    board = create_board()
    rows = config['boardRows']
    cols = config['boardCols']

    # Set start card at row 0, col = cols//2.
    start_col = cols // 2
    for card in board:
        if card.x_index == start_col and card.y_index == 0:
            card.type = 'start'
            break

    # Define goal card positions in the bottom row.
    goal_positions = []
    mid = cols // 2
    if mid - 1 >= 0:
        goal_positions.append((mid - 1, rows - 1))
    goal_positions.append((mid, rows - 1))
    if mid + 1 < cols:
        goal_positions.append((mid + 1, rows - 1))

    if goal_positions:
        gold_pos = random.choice(goal_positions)
        for pos in goal_positions:
            for card in board:
                if card.x_index == pos[0] and card.y_index == pos[1]:
                    if pos == gold_pos:
                        card.type = 'gold'
                    else:
                        card.type = 'coal'
                    # Mark goal cards as hidden initially.
                    card.hidden = True
                    break

    return board
