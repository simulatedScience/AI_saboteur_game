# draw_card.py
"""
This module implements functions to draw saboteur cards to a tkinter GUI.
"""
# standard library imports
import tkinter as tk
# third-party imports

# local imports
from .cards import Card
from .config import GUI_CONFIG

def draw_card(card: Card, parent_widget: tk.Widget) -> tk.Canvas:
    """
    Draw a given card onto the parent widget.
    Use pen to draw the card's path onto a wall-colored background.

    Args:
        card (Card): card to draw
        parent_widget (tk.Widget): parent widget to draw the card onto

    Returns:
        tk.Canvas: canvas with the card drawn. Still needs to be placed.
    """
    # create base card
    card_canvas: tk.Canvas = tk.Canvas(
        parent_widget,
        width=GUI_CONFIG['card_width'],
        height=GUI_CONFIG['card_height'],
        bg=GUI_CONFIG['color_wall']
    )
    # draw paths:
    # if a path exists, draw a point in the middle of the card and connect it to all path-edges.
    # for each dead-end edge, draw a point at the edge's center.
    # consider rotation of the card.
    
    # preliminary calculations
    edge_centers: dict[str, tuple[int, int]] = {
        'top': (GUI_CONFIG['card_width'] / 2, 0),
        'right': (GUI_CONFIG['card_width'], GUI_CONFIG['card_height'] / 2),
        'bottom': (GUI_CONFIG['card_width'] / 2, GUI_CONFIG['card_height']),
        'left': (0, GUI_CONFIG['card_height'] / 2)
    }
    
    # draw paths
    if "path" in card.edges.values():
        # draw center point with path color and width
        card_canvas.create_oval(
            GUI_CONFIG['card_width'] / 2 - GUI_CONFIG['path_width'] / 2,
            GUI_CONFIG['card_height'] / 2 - GUI_CONFIG['path_width'] / 2,
            GUI_CONFIG['card_width'] / 2 + GUI_CONFIG['path_width'] / 2,
            GUI_CONFIG['card_height'] / 2 + GUI_CONFIG['path_width'] / 2,
            fill=GUI_CONFIG['color_path'],
            outline=GUI_CONFIG['color_path']
        )
        # draw lines to path-edges
        for edge, edge_type in card.edges.items():
            if edge_type == "path":
                # draw line from center to edge
                edge_center = edge_centers[edge]
                card_canvas.create_line(
                    GUI_CONFIG['card_width'] / 2,
                    GUI_CONFIG['card_height'] / 2,
                    edge_center[0],
                    edge_center[1],
                    fill=GUI_CONFIG['color_path'],
                    width=GUI_CONFIG['path_width']
                )
    # draw dead-ends
    for edge, edge_type in card.edges.items():
        if edge_type == "dead-end":
            # draw point at edge center
            edge_center = edge_centers[edge]
            card_canvas.create_oval(
                edge_center[0] - GUI_CONFIG['path_width'] / 2,
                edge_center[1] - GUI_CONFIG['path_width'] / 2,
                edge_center[0] + GUI_CONFIG['path_width'] / 2,
                edge_center[1] + GUI_CONFIG['path_width'] / 2,
                fill=GUI_CONFIG['color_dead-end'],
                outline=GUI_CONFIG['color_dead-end']
            )

    return card_canvas
