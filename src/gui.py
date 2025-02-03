# gui.py

"""
This module provides a graphical user interface (GUI) for the Saboteur card game.
The GUI class renders the game board, player info, and current hand.
It allows for card selection, rotation, and placement via the environment's step() method.
The board is draggable via right-click, and valid placement positions are outlined.
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
    GUI class that renders the game board, player info, and current hand.
    Implements card selection, rotation, and placement via the environment's step() method.
    The board is draggable via right-click, and valid placement positions are outlined.
    """
    def __init__(self, env: SaboteurEnv, player_names: list[str] | None = None) -> None:
        self.env: SaboteurEnv = env
        self.env.reset()
        self.root: tk.Tk = tk.Tk()
        self.root.title("Saboteur Card Game")

        # Player names.
        if player_names is None:
            self.player_names: list[str] = [f"Player {i+1}" for i in range(self.env.num_players)]
        else:
            self.player_names = player_names

        # Board offset for dragging.
        self.board_offset_x: int = 0
        self.board_offset_y: int = 0
        self.drag_start_x: int = 0
        self.drag_start_y: int = 0

        # Board extents (will be recalculated in update_board_extents).
        self.min_x: int = 0
        self.max_x: int = 0
        self.min_y: int = 0
        self.max_y: int = 0
        self.board_cols: int = 0
        self.board_rows: int = 0

        # Set an initial canvas size.
        self.canvas_width: int = 800
        self.canvas_height: int = 600
        self.canvas: tk.Canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Selected card (and its index in the current hand).
        self.selected_card: Card | None = None
        self.selected_card_index: int | None = None

        # Bind left-click for card selection/placement and right-click for dragging.
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)

        self.draw()

    def update_board_extents(self) -> None:
        """
        Update the board extents based on placed cards, leaving one extra cell margin around.
        Adjust the canvas size accordingly.
        """
        if self.env.board:
            xs: list[int] = [pos[0] for pos in self.env.board.keys()]
            ys: list[int] = [pos[1] for pos in self.env.board.keys()]
            self.min_x = min(xs) - 1
            self.max_x = max(xs) + 1
            self.min_y = min(ys) - 1
            self.max_y = max(ys) + 1
        else:
            self.min_x, self.max_x, self.min_y, self.max_y = 0, 0, 0, 0

        self.board_cols = self.max_x - self.min_x + 1
        self.board_rows = self.max_y - self.min_y + 1
        new_width: int = GUI_CONFIG['card_margin'] + self.board_cols * (GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin'])
        new_height: int = GUI_CONFIG['card_margin'] + self.board_rows * (GUI_CONFIG['card_height'] + GUI_CONFIG['card_margin']) + 150
        self.canvas_width = new_width
        self.canvas_height = new_height
        self.canvas.config(width=new_width, height=new_height)

    def transform(self, pos: tuple[int, int]) -> tuple[int, int]:
        """
        Convert board coordinates (x, y) to pixel coordinates (taking board offset into account).
        """
        x, y = pos
        pixel_x: int = GUI_CONFIG['card_margin'] + (x - self.min_x) * (GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin']) + self.board_offset_x
        pixel_y: int = GUI_CONFIG['card_margin'] + (y - self.min_y) * (GUI_CONFIG['card_height'] + GUI_CONFIG['card_margin']) + self.board_offset_y
        return pixel_x, pixel_y

    def inverse_transform(self, pixel: tuple[int, int]) -> tuple[int, int]:
        """
        Convert pixel coordinates back to board coordinates (ignoring board offset).
        """
        x_pixel, y_pixel = pixel
        board_x: int = (x_pixel - GUI_CONFIG['card_margin'] - self.board_offset_x) // (GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin']) + self.min_x
        board_y: int = (y_pixel - GUI_CONFIG['card_margin'] - self.board_offset_y) // (GUI_CONFIG['card_height'] + GUI_CONFIG['card_margin']) + self.min_y
        return board_x, board_y

    def draw(self) -> None:
        """
        Redraw the entire canvas: board, valid placement outlines, player info, and hand (or play-again button).
        """
        self.canvas.delete("all")
        self.update_board_extents()
        self.draw_board()
        self.draw_valid_placements()
        self.draw_player_info()
        self.draw_hand()

    def draw_board(self) -> None:
        """
        Draw all cards placed on the board.
        """
        for pos, card in self.env.board.items():
            pixel_x, pixel_y = self.transform(pos)
            card_canvas: tk.Canvas = draw_card(
                card=card,
                parent_widget=self.canvas,
                click_callback=lambda event, c=card: self.on_card_click(event, c, None)
            )
            self.canvas.create_window(pixel_x, pixel_y, window=card_canvas, anchor='nw')
            if card.type == 'goal' and not card.hidden:
                goal_text: str = card.goal_type.upper() if card.goal_type else "GOLD"
                self.canvas.create_text(
                    pixel_x + GUI_CONFIG['card_width'] / 2,
                    pixel_y + GUI_CONFIG['card_height'] / 2,
                    text=goal_text,
                    fill="black",
                    font=(GUI_CONFIG['font'], 12)
                )

    def draw_valid_placements(self) -> None:
        """
        If a card is selected, draw a dashed outline at each board position where the card could be legally placed.
        """
        if self.selected_card is None:
            return
        valid_positions = self.env.get_valid_placements(self.selected_card)
        for pos in valid_positions:
            pixel_x, pixel_y = self.transform(pos)
            self.canvas.create_rectangle(
                pixel_x,
                pixel_y,
                pixel_x + GUI_CONFIG['card_width'],
                pixel_y + GUI_CONFIG['card_height'],
                outline="green",
                dash=(4, 2),
                width=2
            )

    def draw_player_info(self) -> None:
        """
        Draw text showing the current active player.
        """
        info_text: str = f"Current player: {self.player_names[self.env.current_player]} " \
                           f"({self.env.current_player + 1}/{self.env.num_players})"
        self.canvas.create_text(
            10, 10, anchor='nw', text=info_text, font=(GUI_CONFIG['font'], 16), fill="black"
        )

    def draw_hand(self) -> None:
        """
        Draw the current player's hand at the bottom of the canvas.
        If the game is over, display a large "Play Again" button instead.
        """
        if self.env.done:
            center_x: float = self.canvas_width / 2
            center_y: float = self.canvas_height - 75
            play_again_button: tk.Button = tk.Button(
                self.canvas,
                text="Play Again",
                font=(GUI_CONFIG['font'], 24),
                bg="green",
                fg="white",
                command=self.play_again
            )
            self.canvas.create_window(center_x, center_y, window=play_again_button)
            return

        current_hand: list[Card] = self.env.player_hands[self.env.current_player]
        num_cards: int = len(current_hand)
        total_width: float = num_cards * (GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin']) + GUI_CONFIG['card_margin']
        start_x: float = (self.canvas_width - total_width) / 2 + GUI_CONFIG['card_margin']
        y: float = self.canvas_height - 150 + GUI_CONFIG['card_margin']

        for idx, card in enumerate(current_hand):
            card_canvas: tk.Canvas = draw_card(
                card=card,
                parent_widget=self.canvas,
                click_callback=lambda event, c=card, i=idx: self.on_card_click(event, c, i)
            )
            self.canvas.create_window(start_x, y, window=card_canvas, anchor='nw')

            if card.selected:
                self.canvas.create_rectangle(
                    start_x,
                    y,
                    start_x + GUI_CONFIG['card_width'],
                    y + GUI_CONFIG['card_height'],
                    outline=GUI_CONFIG['color_selection_outline'],
                    width=GUI_CONFIG['selection_width']
                )
            start_x += GUI_CONFIG['card_width'] + GUI_CONFIG['card_margin']

    def on_card_click(self, event: tk.Event, card: Card, index: int | None) -> None:
        """
        Handle click events on a card in the hand or on a goal card on the board.
        - For a hidden goal card, temporarily reveal it.
        - For a non-goal card: if already selected, rotate it; otherwise, select it.
        """
        if card.type == 'goal':
            # Clicking on a goal card: if hidden, reveal it briefly.
            if card.hidden:
                card.hidden = False
                self.draw()
                self.root.after(2000, lambda: self.hide_goal_temporarily(card))
            return

        if self.selected_card is not None and self.selected_card == card:
            card.rotate()
        else:
            for c in self.env.player_hands[self.env.current_player]:
                c.selected = False
            card.selected = True
            self.selected_card = card
            self.selected_card_index = index
        self.draw()

    def hide_goal_temporarily(self, card: Card) -> None:
        """
        After a temporary reveal, hide the goal card again unless it has been uncovered by placement.
        """
        # Only re-hide if the card is not reached via a valid path.
        if not self.env.can_reach((card.x, card.y)):
            card.hidden = True
        self.draw()

    def on_click(self, event: tk.Event) -> None:
        """
        Handle left-clicks on the canvas.
        If a card is selected from the hand, attempt to place it at the clicked board position.
        """
        if self.env.done:
            return

        click_x, click_y = event.x, event.y
        board_coord: tuple[int, int] = self.inverse_transform((click_x, click_y))
        if self.selected_card is not None and self.selected_card_index is not None:
            action: tuple[int, tuple[int, int], int] = (
                self.selected_card_index,
                board_coord,
                self.selected_card.rotation
            )
            obs, reward, done, _, _ = self.env.step(action)
            if reward < 0:
                print("Invalid move!")
            self.selected_card = None
            self.selected_card_index = None
            self.draw()

    def on_right_press(self, event: tk.Event) -> None:
        """
        Record the starting position for a board drag (right-click).
        """
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_right_drag(self, event: tk.Event) -> None:
        """
        Update the board offset as the user drags with the right mouse button.
        """
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.board_offset_x += dx
        self.board_offset_y += dy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.draw()

    def play_again(self) -> None:
        """
        Callback for the Play Again button.
        Resets the environment and redraws the GUI.
        """
        self.env.reset()
        self.selected_card = None
        self.selected_card_index = None
        self.board_offset_x = 0
        self.board_offset_y = 0
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
