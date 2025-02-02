# gui.py
"""
GUI for the Saboteur card game.
Implements drawing of the board and hand using the draw_card module.
Allows selection, rotation, and placement of in-hand cards.
"""

import tkinter as tk
from typing import Optional
from .saboteur_env import SaboteurEnv
from .config import GUI_CONFIG, CONFIG
from .cards import Card
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

        self.canvas.bind("<Button-1>", self.on_click)

        # Player hand: create a sample set of five path cards.
        self.player_hand: list[Card] = []
        for _ in range(5):
            # For demonstration, create a path card with two paths.
            card = Card(
                'path',
            )
            card.rotation = 0
            self.player_hand.append(card)
        self.selected_card: Optional[Card] = None  # Track the selected card.

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
            card_canvas = draw_card(card, self.canvas)
            # Place the drawn card canvas on the main canvas.
            self.canvas.create_window(pixel_x, pixel_y, window=card_canvas, anchor='nw')
            # If the card is a goal card and hidden, draw its label.
            if card.type == 'goal' and card.hidden:
                self.canvas.create_text(pixel_x + GUI_CONFIG['card_width'] / 2, pixel_y + GUI_CONFIG['card_height'] / 2,
                                        text="???", fill="black", font=("Arial", 12))
            elif card.type == 'goal' and not card.hidden:
                # Display the goal type.
                goal_text = card.goal_type.upper() if card.goal_type else "GOAL"
                self.canvas.create_text(pixel_x + GUI_CONFIG['card_width'] / 2, pixel_y + GUI_CONFIG['card_height'] / 2,
                                        text=goal_text, fill="black", font=("Arial", 12))

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
            x = start_x
            # Save hand position for hit detection.
            card.hand_pos = (x, y)
            card_canvas = draw_card(card, self.canvas)
            self.canvas.create_window(x, y, window=card_canvas, anchor='nw')
            # If the card is selected, draw an outline.
            if card.selected:
                self.canvas.create_rectangle(
                    x, y, x + GUI_CONFIG['card_width'], y + GUI_CONFIG['card_height'],
                    outline=GUI_CONFIG['selection_outline'], width=3
                )
            start_x += GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin']

    def on_click(self, event: tk.Event) -> None:
        """
        Handle mouse clicks.
        - In the hand area: select a card or, if already selected, rotate it.
        - In the board area: if empty and a card is selected, place that card.
        
        Args:
            event (tk.Event): The click event.
        """
        # Define hand area (the bottom 150px of the canvas).
        hand_area_top = self.canvas_height - 150
        if event.y >= hand_area_top:
            # Process click in hand.
            self.process_hand_click(event.x, event.y)
        else:
            # Process click on the board.
            self.process_board_click(event.x, event.y)

    def process_hand_click(self, x: int, y: int) -> None:
        """
        Process a click in the hand area.
        
        Args:
            x (int): x-coordinate of the click.
            y (int): y-coordinate of the click.
        """
        for card in self.player_hand:
            hx, hy = card.hand_pos
            if hx <= x <= hx + GUI_CONFIG['card_width'] and hy <= y <= hy + GUI_CONFIG['card_height']:
                if card.selected:
                    # Clicking an already selected card rotates it.
                    card.rotate()
                else:
                    # Select this card and deselect all others.
                    for other in self.player_hand:
                        other.selected = False
                    card.selected = True
                    self.selected_card = card
                self.draw()
                return

    def process_board_click(self, x: int, y: int) -> None:
        """
        Process a click in the board area.
        If the click is on an empty board cell and a card is selected, place the card there.
        
        Args:
            x (int): x-coordinate of the click.
            y (int): y-coordinate of the click.
        """
        board_coord = self.inverse_transform((x, y))
        # Check if a card is already present at that board coordinate.
        if board_coord in self.env.board:
            # Optionally, handle clicks on already placed cards (e.g., reveal goals).
            card = self.env.board[board_coord]
            if card.type == 'goal' and card.hidden:
                card.hidden = False
                self.draw()
                self.root.after(2000, lambda c=card: self.hide_goal(c))
            return
        # If a card is selected from the hand, place it.
        if self.selected_card is not None:
            # Remove the card from hand and place it on the board.
            self.selected_card.x, self.selected_card.y = board_coord
            self.env.board[board_coord] = self.selected_card
            # Deselect and remove from hand.
            self.selected_card.selected = False
            self.player_hand.remove(self.selected_card)
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
