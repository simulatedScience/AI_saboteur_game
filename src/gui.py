"""
Updated GUI to use env.place_card(...) for valid card placement.

We assume the saboteur_env.py now has a method env.place_card(card, (x, y)) -> bool.
The user can select a card in their hand, click on the board to attempt placement, and if successful, the card is removed from the hand.
"""

# Standard library imports
import random
import tkinter as tk

# Third-party imports

# Local imports
from .saboteur_env import SaboteurEnv
from .config import GUI_CONFIG
from .cards import Card, get_random_edges, calculate_connections
from .draw_card import draw_card

class SaboteurGUI:
    """
    GUI class that renders the game board and player hand,
    and implements card selection, rotation, and placement.

    Updated to call env.place_card(...) when the user clicks.
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

        # create a sample set of five random path cards for the player's hand.
        self.player_hand: list[Card] = []
        for _ in range(5):
            card = Card('path')
            card.rotation = 0
            self.player_hand.append(card)
        self.selected_card: Card | None = None

        self.canvas.bind("<Button-1>", self.on_click)
        self.draw()

    def transform(self, pos: tuple[int, int]) -> tuple[int, int]:
        """
        Transform board coordinate pos (x, y) to pixel coordinates.
        """
        x, y = pos
        pixel_x = GUI_CONFIG['card_margin'] + (x - self.min_x) * (GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin'])
        pixel_y = GUI_CONFIG['card_margin'] + (y - self.min_y) * (GUI_CONFIG['card_height'] + GUI_CONFIG['card_margin'])
        return pixel_x, pixel_y

    def inverse_transform(self, pixel: tuple[int, int]) -> tuple[int, int]:
        """
        Convert pixel coordinates back to board coordinates.
        """
        x_pixel, y_pixel = pixel
        board_x = (x_pixel - GUI_CONFIG['card_margin']) // (GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin']) + self.min_x
        board_y = (y_pixel - GUI_CONFIG['card_margin']) // (GUI_CONFIG['card_height'] + GUI_CONFIG['card_margin']) + self.min_y
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
                click_callback=lambda event, c=card: self.on_card_click(event, c)
            )
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
                click_callback=lambda event, c=card: self.on_card_click(event, c)
            )
            self.canvas.create_window(start_x, y, window=card_canvas, anchor='nw')

            if card.selected:
                # highlight the card with a rectangle.
                self.canvas.create_rectangle(
                    start_x,
                    y,
                    start_x + GUI_CONFIG['card_width'],
                    y + GUI_CONFIG['card_height'],
                    outline=GUI_CONFIG['color_selection_outline'],
                    width=GUI_CONFIG['selection_width'],
                )

            start_x += GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin']

    def on_card_click(self, event, card: Card) -> None:
        """
        Handle click events on a card in the hand.
        If card was already selected, rotate it.
        If not, deselect all other cards and select this one.
        """
        if card.type == 'goal':
            if card.hidden:
                # show the goal card for 2 seconds
                card.hidden = False
                self.draw()
                self.root.after(2000, lambda: self.on_card_click(event, card))
            else:
                # hide the goal card again
                card.hidden = True
                self.draw()
            return
        if card.selected:
            card.rotate()
        else:
            for other in self.player_hand:
                other.selected = False
            card.selected = True
            self.selected_card = card
        self.draw()

    def on_click(self, event) -> None:
        """
        Handle click events on the canvas.
        If we have a selected card and the user clicks on an empty board space, attempt placement.
        """
        click_x, click_y = event.x, event.y
        board_coord = self.inverse_transform((click_x, click_y))

        # If we have a selected card, try placing it.
        if self.selected_card is not None:
            success = self.env.place_card(self.selected_card, board_coord)
            if success:
                # remove from hand
                self.player_hand.remove(self.selected_card)
                self.selected_card = None
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
