# draw_card.py
"""
This module implements functions to draw Saboteur cards to a tkinter GUI.
"""

# Standard library imports
import tkinter as tk

# Third-party imports

# Local imports
from .cards import Card
from .config import GUI_CONFIG

def draw_card(card: Card, parent_widget: tk.Widget, click_callback: callable = None) -> tk.Canvas:
    """
    Draw a given card onto the parent widget.
    Uses drawing primitives to render the card's path onto a wall-colored background.
    For goal cards that are hidden, draws a blank (back) with no markings.

    Args:
        card (Card): Card to draw.
        parent_widget (tk.Widget): Parent widget to draw the card onto.
        click_callback (callable): Callback function for mouse clicks on the card.

    Returns:
        tk.Canvas: Canvas with the card drawn (to be placed later).
    """
    # For goal cards that are hidden, return a blank canvas.
    if card.type == "goal":
        card_canvas: tk.Canvas = draw_goal_card(
            card=card,
            parent_widget=parent_widget,
            click_callback=click_callback,
        )
        return card_canvas

    # elif card.type in ("path", "start"):
    card_canvas: tk.Canvas = draw_path_card(
        card=card,
        parent_widget=parent_widget,
        click_callback=click_callback,
    )
    return card_canvas

def draw_goal_card(
            card: Card,
            parent_widget: tk.Widget,
            click_callback: callable = None,
        ) -> tk.Canvas:
    """
    Draw a goal card onto the parent widget.
    This function draws a blank card with a question mark on it.

    Args:
        card (Card): Card to draw.
        parent_widget (tk.Widget): Parent widget to draw the card onto.
        click_callback (callable): Callback function for mouse clicks on the card.

    Returns:
        tk.Canvas: Canvas with the card drawn (to be placed later).
    """
    if card.hidden:
        card_canvas: tk.Canvas = tk.Canvas(
                parent_widget,
                width=GUI_CONFIG['card_width'],
                height=GUI_CONFIG['card_height'],
                bg=GUI_CONFIG['color_wall']
            )
            # add question mark on card
        card_canvas.create_text(
                GUI_CONFIG['card_width'] / 2,
                GUI_CONFIG['card_height'] / 2,
                text="?",
                font=(GUI_CONFIG['font'], 24, "bold"),
                fill=GUI_CONFIG['color_goal_hidden']
            )
        card_canvas.bind("<Button-1>", click_callback)
    else:
        card_canvas = draw_path_card(
            card=card,
            parent_widget=parent_widget,
            click_callback=click_callback
        )
        # draw gold or coal on card
        is_coal = "wall" in card.edges.values()
        goal_color = GUI_CONFIG['color_goal_coal'] if is_coal else GUI_CONFIG['color_goal_gold']
        # draw circle at 0.8 path_width in center
        resource_size: float = 0.5 * GUI_CONFIG['path_width']
        card_canvas.create_oval(
            GUI_CONFIG['card_width'] / 2 - resource_size,
            GUI_CONFIG['card_height'] / 2 - resource_size,
            GUI_CONFIG['card_width'] / 2 + resource_size,
            GUI_CONFIG['card_height'] / 2 + resource_size,
            fill=goal_color,
            outline=goal_color
        )
    return card_canvas

def draw_path_card(
            card: Card,
            parent_widget: tk.Widget,
            click_callback: callable = None
        ) -> tk.Canvas:
    """
    Draw a path card onto the parent widget.
    This function draws the card's path and dead-end edges onto a wall-colored background.
    
    Args:
        card (Card): Card to draw.
        parent_widget (tk.Widget): Parent widget to draw the card onto.
        click_callback (callable): Callback function for mouse clicks on the card.
        
    Returns:
        tk.Canvas: Canvas with the card drawn (to be placed later).
    """
    # Create path card
    card_canvas = tk.Canvas(
        parent_widget,
        width=GUI_CONFIG['card_width'],
        height=GUI_CONFIG['card_height'],
        bg=GUI_CONFIG['color_wall']
    )
    card_canvas.bind("<Button-1>", click_callback)

    # Preliminary calculations for edge centers.
    edge_centers: dict[str, tuple[int, int]] = {
        'top': (GUI_CONFIG['card_width'] // 2, 0),
        'right': (GUI_CONFIG['card_width'], GUI_CONFIG['card_height'] // 2),
        'bottom': (GUI_CONFIG['card_width'] // 2, GUI_CONFIG['card_height']),
        'left': (0, GUI_CONFIG['card_height'] // 2)
    }
    # Draw paths: if any edge is a path, draw a central point and connect it to each "path" edge.
    if "path" in card.edges.values():
        card_canvas.create_oval(
            GUI_CONFIG['card_width'] / 2 - GUI_CONFIG['path_width'] / 2,
            GUI_CONFIG['card_height'] / 2 - GUI_CONFIG['path_width'] / 2,
            GUI_CONFIG['card_width'] / 2 + GUI_CONFIG['path_width'] / 2,
            GUI_CONFIG['card_height'] / 2 + GUI_CONFIG['path_width'] / 2,
            fill=GUI_CONFIG['color_path'],
            outline=GUI_CONFIG['color_path']
        )
        # Draw lines from the center to each path edge.
        for edge, edge_type in card.edges.items():
            if edge_type == "path":
                edge_center = edge_centers[edge]
                card_canvas.create_line(
                    GUI_CONFIG['card_width'] / 2,
                    GUI_CONFIG['card_height'] / 2,
                    edge_center[0],
                    edge_center[1],
                    fill=GUI_CONFIG['color_path'],
                    width=GUI_CONFIG['path_width']
                )
    # Draw dead-ends: for each dead-end edge, draw a small circle at that edge's center.
    for edge, edge_type in card.edges.items():
        if edge_type == "dead-end":
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
