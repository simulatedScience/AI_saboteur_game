
# config.py
GUI_CONFIG = {
    # Card size and margins (in pixels)
    'card_width': 80,
    'card_height': 120,
    'card_margin': 1,
    'path_width': 20,
    'selection_width': 5,
    # Canvas settings
    "zoom": 1,
    "max_canvas_width": 1200,
    "max_canvas_height": 800,
    # AI-related settings
    'ai_delay': 200,  # Delay in milliseconds for AI moves
    "auto_start": True,
    # Font
    'font': 'Roboto',
    'color_text_bright': "#eeeeee",       # white
    'color_text_dark': "#333333",         # dark gray
    # Colors
    'color_goal_hidden': "#999999",       # medium gray
    'color_goal_gold': "#ccaa00",         # gold
    'color_goal_coal': "#222222",         # almost black
    'color_wall': "#444444",              # dark gray
    'color_path': "#eeddaa",              # dark beige
    'color_dead-end': "#eeddaa",          # dark beige
    'color_selection_outline': "#5588ff", # blue
    # button colors
    'color_play_again': "#00aa00",        # green
    'color_play_again_active': "#44bb44", # light green
    'color_skip': "#dd7700",              # orange
    'color_skip_active': "#ee8822",       # light orange
}

CONFIG = {
    'hand_size': 6,
    'deck': 'src/assets/deck_standard.json',
    # "AI_TYPES": ["human", "rule-based", "rule-based", "rule-based"],
    # "AI_TYPES": ["RL-agent", "RL-agent", "RL-agent"],
    "AI_TYPES": ["20250212_002339", "20250212_002339", "20250212_002339", "20250212_002339"],
    # "AI_TYPES": ["rule-based", "rule-based", "rule-based", "rule-based"],
    # "AI_TYPES": ["random", "rule-based"],
}
CONFIG['num_players'] = len(CONFIG['AI_TYPES'])

# AI configuration.
AI_CONFIG = {
    'learning_rate': 0.003,  # Learning rate
    'gamma': 0.99,  # Discount factor
    'timesteps': 100_000_000,  # Number of timesteps to train
    'batch_size': 512,  # Batch size
    'n_envs': 24,  # Number of parallel environments
    'device': 'cpu', # Device to use for training ('cuda' or 'cpu')
    'n_eval_episodes': 3,  # Number of episodes to evaluate
    'n_steps': 100*24,  # max. Number of steps per environment (must be multiple of n_envs)
    'log_interval': 1,  # Log interval

    'dist_reward_scale': 0.0, # max intermediate reward
    'mask_dropout_prob': 0.1, # mask dropout probability
    # 'dqn_hidden_layers': (256, 256) # hidden layers in NN
    "final_reward_winner": 10,  # The winning player receives 4 points; subsequent players get one less each.
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