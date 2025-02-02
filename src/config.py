# config.py
GUI_CONFIG = {
    # Card size and margins (in pixels)
    'card_width': 80,
    'card_height': 120,
    'card_margin': 10,
    'path_width': 20,
    # Colors (use softer, less bold shades)
    'color_start': "#ccbb88",        # soft blue
    'color_goal_hidden': "#666666",    # light gray
    'color_goal_gold': "#ddbb00",      # gold
    'color_goal_coal': "#222222",      # almost black
    'color_wall': "#444444",           # dark gray
    'color_path': "#eeddaa",           # light green
    'color_dead-end': "#eeddaa",       # light pink
    'color_hand': "#777777",           # light gray
}

CONFIG = {
    'boardCols': 8,
    'boardRows': 5,
    'numPlayers': 4,  # default number of players
}

def check_config() -> bool:
    """Check if the configuration is valid."""
    valid_config: bool = True
    # Check positivity of values
    for config_key in ('card_width', 'card_height', 'card_margin', 'path_width'):
        if not isinstance(GUI_CONFIG[config_key], int):
            print(f"Invalid configuration: {config_key} must be a positive integer.")
            valid_config = False
        if GUI_CONFIG[config_key] <= 0:
            print(f"Invalid configuration: {config_key} must be positive.")
            valid_config = False
    # check valid path width
    if GUI_CONFIG['path_width'] >= min(GUI_CONFIG['card_width'], GUI_CONFIG['card_height'])/2:
        print(f"Invalid configuration: path width must be less than half the card width and height.")
        valid_config = False

    return True

assert check_config()

if __name__ == "__main__":
    print("Configuration is valid.")