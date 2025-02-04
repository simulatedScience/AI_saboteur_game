
# config.py
GUI_CONFIG = {
    # Card size and margins (in pixels)
    'card_width': 50,
    'card_height': 60,
    'card_margin': 1,
    'path_width': 15,
    'selection_width': 5,
    # AI-related settings
    'ai_delay': 200,  # Delay in milliseconds for AI moves
    "auto_start": True,
    # Font
    'font': 'Roboto',
    # Colors (use softer, less bold shades)
    'color_start': "#ccbb88",             # light brown
    'color_goal_hidden': "#999999",       # medium gray
    'color_goal_gold': "#ccaa00",         # gold
    'color_goal_coal': "#222222",         # almost black
    'color_wall': "#444444",              # dark gray
    'color_path': "#eeddaa",              # dark beige
    'color_dead-end': "#eeddaa",          # dark beige
    'color_hand': "#777777",              # light gray
    'color_selection_outline': "#5588ff", # blue
}

CONFIG = {
    'num_players': 4,  # default number of players
    'hand_size': 6,
    'deck': 'src/assets/deck_standard.json',
    "AI_TYPES": ["human", "rule-based", "rule-based", "rule-based"],
    "AI_TYPES": ["rule-based", "rule-based", "rule-based", "rule-based"],
    # "AI_TYPES": ["random", "rule-based"],
}

# AI configuration.
AI_CONFIG = {
    "final_reward_winner": 4,  # The winning player receives 4 points; subsequent players get one less each.
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

    return valid_config

assert check_config()

if __name__ == "__main__":
    print("Configuration is valid.")