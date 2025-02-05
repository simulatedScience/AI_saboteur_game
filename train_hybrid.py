# agents/train_hybrid.py
"""
Training script for the Saboteur RL agent using Stable-Baselines3’s PPO.
This script uses the SaboteurHybridWrapper and logs training via TensorBoard.
Each training run creates a new folder (named with the current datetime) to save the model,
checkpoints, and a training_info.md file with hyperparameters and configuration.
"""

import os
from datetime import datetime
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
# import CheckPointCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from src.agents.saboteur_hybrid_wrapper import SaboteurHybridWrapper, STATE_SIZE
from src.config import AI_CONFIG, CONFIG

def create_run_folder() -> str:
    """
    Create and return a new folder for this training run.
    
    Returns:
        str: Path to the run folder.
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join("training_runs", now)
    os.makedirs(folder, exist_ok=True)
    return folder

def save_training_info(folder: str) -> None:
    """
    Save training configuration and hyperparameters to a markdown file.
    
    Args:
        folder (str): The folder to save the file in.
    """
    info_text = f"""# Training Information

**Run Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Environment Configuration:**
{CONFIG}

**AI Configuration:**
{AI_CONFIG}

State Dimension: {STATE_SIZE}
"""
    with open(os.path.join(folder, "training_info.md"), "w") as f:
        f.write(info_text)

def action_mask_fn(env: gym.Env, obs: dict) -> list:
    """
    (Optional) Compute an action mask.
    For the hybrid action space, we can use the environment’s valid placements to mask invalid actions.
    For simplicity, here we return a mask that always allows all actions.
    
    Args:
        env (gym.Env): The environment.
        obs (dict): The observation.
        
    Returns:
        list: A binary list of length equal to the discrete component of the action space.
    """
    # For hybrid action space, we assume no masking at this time.
    # (In a full implementation, compute a mask over all (card, coord, orient) combinations.)
    return [1] * env.action_space["card"].n * 1 * env.action_space["orient"].n

def train():
    run_folder = create_run_folder()
    save_training_info(run_folder)
    env = SaboteurHybridWrapper()
    # Wrap the environment with an action masker if desired (here, we use a dummy mask function).
    env = ActionMasker(env, action_mask_fn)
    model = MaskablePPO(
        "MlpPolicy",  # Policy that supports dict observations
        env,
        verbose=1,
        tensorboard_log=os.path.join(run_folder, "tensorboard"),
        **AI_CONFIG
    )
    eval_env = SaboteurHybridWrapper()
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=run_folder,
        log_path=run_folder,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=run_folder,
        name_prefix="checkpoint"
    )
    total_timesteps = AI_CONFIG.get("timesteps", 100000)
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])
    model.save(os.path.join(run_folder, "trained_model"))
    env.close()

if __name__ == "__main__":
    train()
