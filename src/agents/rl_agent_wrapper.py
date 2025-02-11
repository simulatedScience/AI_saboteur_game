import os
import numpy as np
from sb3_contrib import MaskablePPO
from ..saboteur_env import SaboteurEnv
from ..agents.saboteur_discrete_wrapper import SaboteurDiscreteWrapper, COORD_LOW, COORD_HIGH, COORD_RES

class RLAgentWrapper:
    def __init__(self, model_path: str, env: SaboteurEnv) -> None:
        """
        Initialize the RLAgentWrapper by loading the model and creating a discrete wrapper.
        
        Args:
            model_path (str): Path to the trained_model.zip file.
            env (SaboteurEnv): The Saboteur environment instance.
        """
        if not os.path.exists(model_path):
            raise ValueError(f"RL model path {model_path} does not exist.")
        self.model = MaskablePPO.load(model_path)
        # Create a discrete wrapper using the same underlying env.
        self.wrapper = SaboteurDiscreteWrapper()
        self.wrapper.env = env  # Use the same env instance.
        self.step_size = (COORD_HIGH - COORD_LOW) / (COORD_RES - 1)

    def act(self, player_index: int) -> tuple[int, tuple[float, float], int]:
        """
        Select an action using the loaded RL model.
        
        Obtains the state vector from the discrete wrapper, then uses the model to predict an action.
        The discrete action output (a vector of 4 integers) is converted into an action tuple.
        Importantly, we then “snap” the chosen continuous coordinate to the closest valid placement
        using the same _snap_to_valid() method as in training.
        
        Args:
            player_index (int): Index of the current player.
            
        Returns:
            tuple[int, tuple[float, float], int]: (card_index, (x, y), orientation)
        """
        # Get the current state vector from the wrapper.
        state = self.wrapper._get_state(player_index)
        # Predict action using the RL model.
        action, _ = self.model.predict(state, deterministic=True)
        card_index = int(action[0])
        x_index = int(action[1])
        y_index = int(action[2])
        orientation = int(action[3])
        # Convert discrete indices for x and y into continuous coordinates.
        x_val = COORD_LOW + x_index * self.step_size
        y_val = COORD_LOW + y_index * self.step_size
        hand = self.wrapper.env.player_hands[player_index]
        if card_index >= len(hand):
            return (-1, (0.0, 0.0), 0)
        selected_card = hand[card_index]
        snapped_coord = self.wrapper._snap_to_valid(selected_card, (x_val, y_val), orientation)
        if snapped_coord == (0.0, 0.0):
            return (-1, (0.0, 0.0), 0)
        return (card_index, snapped_coord, orientation)
