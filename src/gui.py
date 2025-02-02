"""
GUI for the Saboteur card game.
Implements drawing of the board and hand using the draw_card module.
Allows selection, rotation, and placement of in-hand cards.
Cards can only be placed adjacent to existing cards.
After placement, the hand is replenished with a new random card.
"""

# Standard library imports
import random

# Third-party imports
import tkinter as tk

# Local imports
from .saboteur_env import SaboteurEnv
from .config import GUI_CONFIG
from .cards import Card, get_random_edges, calculate_connections
from .draw_card import draw_card

class SaboteurGUI:
    """
    GUI class that renders the game board and player hand,
    and implements card selection, rotation, and placement.
    """
    def __init__(self, env: SaboteurEnv) -> None:
        self.env: SaboteurEnv = env
        self.env.reset()
        self.root: tk.Tk = tk.Tk()
        self.root.title("Saboteur Card Game")

        # Calculate board extents based on existing card positions.
        xs = [pos[0] for pos in self.env.board.keys()]
        ys = [pos[1] for pos in self.env.board.keys()]
        self.min_x: int = min(xs)
        self.max_x: int = max(xs)
        self.min_y: int = min(ys)
        self.max_y: int = max(ys)
        self.board_cols: int = self.max_x - self.min_x + 1
        self.board_rows: int = self.max_y - self.min_y + 1

        self.canvas_width: int = GUI_CONFIG['card_margin'] + self.board_cols * (GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin'])
        # Reserve extra space at the bottom for the hand.
        self.canvas_height: int = GUI_CONFIG['card_margin'] + self.board_rows * (GUI_CONFIG['card_height'] + GUI_CONFIG['card_margin']) + 150

        self.canvas: tk.Canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        # self.canvas.bind("<Button-1>", self.on_click)

        # Player hand: create a sample set of five random path cards.
        self.player_hand: list[Card] = []
        for _ in range(5):
            card = Card('path')
            card.rotation = 0
            self.player_hand.append(card)
        self.selected_card: Card | None = None  # Track the selected card.

        self.draw()

    def transform(self, pos: tuple[int, int]) -> tuple[int, int]:
        """
        Transform board coordinate pos (x, y) to pixel coordinates.

        Args:
            pos (tuple[int, int]): The board coordinate.

        Returns:
            tuple[int, int]: The (x, y) pixel coordinate.
        """
        x, y = pos
        pixel_x = GUI_CONFIG['card_margin'] + (x - self.min_x) * (GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin'])
        pixel_y = GUI_CONFIG['card_margin'] + (y - self.min_y) * (GUI_CONFIG['card_height'] + GUI_CONFIG['card_margin'])
        return pixel_x, pixel_y

    def inverse_transform(self, pixel: tuple[int, int]) -> tuple[int, int]:
        """
        Convert pixel coordinates back to board coordinates.

        Args:
            pixel (tuple[int, int]): The (x, y) pixel coordinate.

        Returns:
            tuple[int, int]: The board coordinate.
        """
        x_pixel, y_pixel = pixel
        board_x = round((x_pixel - GUI_CONFIG['card_margin']) / (GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin']) + self.min_x)
        board_y = round((y_pixel - GUI_CONFIG['card_margin']) / (GUI_CONFIG['card_height'] + GUI_CONFIG['card_margin']) + self.min_y)
        return board_x, board_y

    def draw(self) -> None:
        """
        Redraw the entire canvas: the board and the player's hand.
        """
        self.canvas.delete("all")
        self.draw_board()
        self.draw_hand()

    def draw_board(self) -> None:
        """
        Draw all cards on the board using the draw_card module.
        """
        for pos, card in self.env.board.items():
            pixel_x, pixel_y = self.transform(pos)
            card_canvas = draw_card(
                card=card,
                parent_widget=self.canvas,
                click_callback=self.on_click)
            self.canvas.create_window(pixel_x, pixel_y, window=card_canvas, anchor='nw')
            if card.type == 'goal':
                if card.hidden:
                    # For a goal card with a blank back, nothing else is drawn.
                    pass
                else:
                    goal_text = card.goal_type.upper() if card.goal_type else "GOLD"
                    self.canvas.create_text(
                        pixel_x + GUI_CONFIG['card_width'] / 2,
                        pixel_y + GUI_CONFIG['card_height'] / 2,
                        text=goal_text, fill="black", font=(GUI_CONFIG['font'], 12)
                    )

    def draw_hand(self) -> None:
        """
        Draw the player's hand at the bottom of the canvas.
        Uses the draw_card module to draw each card.
        """
        num_cards = len(self.player_hand)
        total_width = num_cards * (GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin']) + GUI_CONFIG['card_margin']
        start_x = (self.canvas_width - total_width) / 2 + GUI_CONFIG['card_margin']
        y = self.canvas_height - 150 + GUI_CONFIG['card_margin']

        for card in self.player_hand:
            card_canvas = draw_card(
                card=card,
                parent_widget=self.canvas,
                click_callback=lambda event, card=card: self.on_card_click(event, card)
            )
            self.canvas.create_window(start_x, y, window=card_canvas, anchor='nw')
            if card.selected:
                self.canvas.create_rectangle(
                    start_x,
                    y,
                    start_x + GUI_CONFIG['card_width'] + GUI_CONFIG['selection_width'],
                    y + GUI_CONFIG['card_height'] + GUI_CONFIG['selection_width'],
                    outline=GUI_CONFIG['color_selection_outline'],
                    width=GUI_CONFIG['selection_width'],
                )
            start_x += GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin']

    def on_card_click(self, event, card):
        """
        Handle click events on a card.
        If card was already selected, rotate it.
        If not, deselect all other cards and select this one.

        Args:
            event: The event object containing the click coordinates.
            card: The card that was clicked.
        """
        if card.selected:
            card.rotate()
        else:
            for other in self.player_hand:
                other.selected = False
            card.selected = True
            self.selected_card = card
        self.draw()  # Redraw the canvas to reflect the selection

    def on_click(self, event):
        """
        Handle click events on the canvas.

        Args:
            event: The event object containing the click coordinates.
        """
        click_x, click_y = event.x, event.y

        # Check if the click is within the bounds of any card in the player's hand
        for card in self.player_hand:
            card_x, card_y = self.transform((card.x, card.y))  # Assuming card has a position attribute
            card_width = GUI_CONFIG['card_width']
            card_height = GUI_CONFIG['card_height']

            if card_x <= click_x <= card_x + card_width and card_y <= click_y <= card_y + card_height:
                self.selected_card = card
                break
        else:
            self.selected_card = None  # Deselect if no card is clicked

        self.draw()  # Redraw the canvas to reflect the selection

    def process_hand_click(self, x: int, y: int) -> None:
        """
        Process a click in the hand area.

        Args:
            x (int): x-coordinate of the click.
            y (int): y-coordinate of the click.
        """
        for card in self.player_hand:
            hx, hy = card.hand_pos  # type: ignore
            if hx <= x <= hx + GUI_CONFIG['card_width'] and hy <= y <= hy + GUI_CONFIG['card_height']:
                if card.selected:
                    card.rotate()
                else:
                    for other in self.player_hand:
                        other.selected = False
                    card.selected = True
                    self.selected_card = card
                self.draw()
                return

    def is_adjacent(self, board_coord: tuple[int, int]) -> bool:
        """
        Check if the given board coordinate is adjacent (cardinally)
        to any already placed card.

        Args:
            board_coord (tuple[int, int]): The board coordinate to check.

        Returns:
            bool: True if adjacent, False otherwise.
        """
        x, y = board_coord
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return any(neighbor in self.env.board for neighbor in neighbors)

    def process_board_click(self, x: int, y: int) -> None:
        """
        Process a click in the board area.
        If the click is on an empty board cell that is adjacent to an existing card and a card is selected, place it.

        Args:
            x (int): x-coordinate of the click.
            y (int): y-coordinate of the click.
        """
        board_coord = self.inverse_transform((x, y))
        if board_coord in self.env.board:
            card = self.env.board[board_coord]
            if card.type == 'goal' and card.hidden:
                card.hidden = False
                self.draw()
                self.root.after(2000, lambda c=card: self.hide_goal(c))
            return
        if not self.is_adjacent(board_coord):
            # Only allow placement adjacent to an existing card.
            return
        if self.selected_card is not None:
            self.selected_card.x, self.selected_card.y = board_coord
            self.env.board[board_coord] = self.selected_card
            # After placement, remove the card from the hand and add a new random card.
            self.player_hand.remove(self.selected_card)
            new_card = Card('path')
            self.player_hand.append(new_card)
            self.selected_card.selected = False
            self.selected_card = None
            self.draw()

    def hide_goal(self, card: Card) -> None:
        """
        Hide a revealed goal card after 2 seconds.

        Args:
            card (Card): The goal card to hide.
        """
        card.hidden = True
        self.draw()

    def run(self) -> None:
        """
        Start the Tkinter main loop.
        """
        self.root.mainloop()

if __name__ == "__main__":
    env = SaboteurEnv()
    gui = SaboteurGUI(env)
    gui.run()
