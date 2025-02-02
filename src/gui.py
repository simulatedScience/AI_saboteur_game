# gui.py
import tkinter as tk
from .saboteur_env import SaboteurEnv
from .config import gui_config, config
from .cards import Card

class SaboteurGUI:
    """
    A basic GUI for the Saboteur game using Tkinter.
    The GUI taps directly into the gymnasium environment for game logic.
    It now includes:
      - A game board with a start card and goal cards.
      - Clicking a goal card reveals it for 2 seconds.
      - A player hand at the bottom where cards can be rotated.
    """
    def __init__(self, env: SaboteurEnv):
        self.env = env

        # Calculate board canvas dimensions from configuration.
        self.board_width = config['boardCols'] * (gui_config['cardWidth'] + gui_config['cardMargin']) + gui_config['cardMargin']
        self.board_height = config['boardRows'] * (gui_config['cardHeight'] + gui_config['cardMargin']) + gui_config['cardMargin']

        # Define hand area dimensions (one row of cards).
        self.hand_height = gui_config['cardHeight'] + 2 * gui_config['cardMargin']
        self.canvas_width = self.board_width
        self.canvas_height = self.board_height + self.hand_height

        self.root = tk.Tk()
        self.root.title("Saboteur Card Game")

        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        # Bind mouse click events.
        self.canvas.bind("<Button-1>", self.on_click)

        # Player hand: For demo purposes, generate 5 random path cards.
        self.player_hand = []
        for i in range(5):
            # Hand cards don't have a board position; we'll assign x,y later.
            card = Card('path', 0, 0)
            card.rotation = 0
            self.player_hand.append(card)

        self.draw()

    def draw(self):
        """Redraw the entire canvas: board and player hand."""
        self.canvas.delete("all")
        self.draw_board()
        self.draw_hand()

    def draw_board(self):
        """Draw the game board based on the current state in the environment."""
        board = self.env.board
        for card in board:
            x = gui_config['cardMargin'] + card.x_index * (gui_config['cardWidth'] + gui_config['cardMargin'])
            y = gui_config['cardMargin'] + card.y_index * (gui_config['cardHeight'] + gui_config['cardMargin'])
            # Determine fill color and label based on card type.
            if card.type == 'start':
                fill_color = "blue"
                text = "START"
            elif card.type in ['gold', 'coal']:
                if card.hidden:
                    fill_color = "gray"  # face-down goal card
                    text = "???"
                else:
                    fill_color = "yellow" if card.type == 'gold' else "brown"
                    text = card.type.upper()
            elif card.type == 'path':
                fill_color = "green"
                text = "PATH"
            elif card.type == 'dead-end':
                fill_color = "red"
                text = "DEAD"
            else:
                fill_color = "black"
                text = card.type.upper()

            self.canvas.create_rectangle(x, y, x + gui_config['cardWidth'], y + gui_config['cardHeight'],
                                         fill=fill_color, outline="black")
            self.canvas.create_text(x + gui_config['cardWidth']/2, y + gui_config['cardHeight']/2,
                                    text=text, fill="white", font=("Arial", 12, "bold"))

    def draw_hand(self):
        """Draw the player's hand in the hand area at the bottom of the canvas."""
        num_cards = len(self.player_hand)
        # Calculate spacing: center the hand on the canvas.
        total_width = num_cards * (gui_config['cardWidth'] + gui_config['cardMargin']) + gui_config['cardMargin']
        start_x = (self.canvas_width - total_width) / 2 + gui_config['cardMargin']
        y = self.board_height + gui_config['cardMargin']

        for i, card in enumerate(self.player_hand):
            x = start_x + i * (gui_config['cardWidth'] + gui_config['cardMargin'])
            # Save the hand card position for hit detection.
            card.hand_pos = (x, y)
            # For rotated cards, we simulate the effect by appending "(180°)" in the label.
            text = f"{card.type.upper()}\n({card.rotation}°)"
            self.canvas.create_rectangle(x, y, x + gui_config['cardWidth'], y + gui_config['cardHeight'],
                                         fill="purple", outline="black")
            self.canvas.create_text(x + gui_config['cardWidth']/2, y + gui_config['cardHeight']/2,
                                    text=text, fill="white", font=("Arial", 12, "bold"))

    def on_click(self, event):
        """
        Handle mouse clicks on the canvas.
        Distinguish between clicks on the board and clicks on the hand area.
        - On board: if a hidden goal card is clicked, reveal it for 2 seconds.
        - On hand: toggle card rotation.
        """
        # Check if click is in the board area.
        if event.y < self.board_height:
            # Determine grid indices.
            col = (event.x - gui_config['cardMargin']) // (gui_config['cardWidth'] + gui_config['cardMargin'])
            row = (event.y - gui_config['cardMargin']) // (gui_config['cardHeight'] + gui_config['cardMargin'])
            # Look for a card at that grid position.
            for card in self.env.board:
                if card.x_index == col and card.y_index == row:
                    # If this is a goal card and is hidden, reveal it temporarily.
                    if card.type in ['gold', 'coal'] and card.hidden:
                        card.hidden = False
                        self.draw()
                        # After 2 seconds, hide it again.
                        self.root.after(2000, lambda c=card: self.hide_goal(c))
                    break
        else:
            # Click is in the hand area.
            # Check each hand card.
            for card in self.player_hand:
                x, y = card.hand_pos
                if x <= event.x <= x + gui_config['cardWidth'] and y <= event.y <= y + gui_config['cardHeight']:
                    # Toggle rotation between 0 and 180.
                    card.rotation = 180 if card.rotation == 0 else 0
                    self.draw()
                    break

    def hide_goal(self, card):
        """Hide the goal card (set hidden flag to True) and redraw."""
        card.hidden = True
        self.draw()

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()

# To run the GUI directly:
if __name__ == "__main__":
    env = SaboteurEnv()
    gui = SaboteurGUI(env)
    gui.run()
