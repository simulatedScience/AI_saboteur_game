# train_discrete.py
"""
Training script for the Saboteur RL agent using Stable-Baselines3's MaskablePPO.
This version uses a vectorized environment (via make_vec_env) for parallelization,
which should greatly increase training speed (from ~60fps to >10000fps).
TensorBoard logging is enabled and a new run folder is created for each training run.
"""

import os
from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.wrappers import ActionMasker

from src.agents.mask_function import mask_func
from src.agents.saboteur_discrete_wrapper import SaboteurDiscreteWrapper, STATE_SIZE, COORD_RES
from src.config import AI_CONFIG, CONFIG

def create_run_folder() -> str:
    """
    Create a new folder for the training run based on the current datetime.
    
    Returns:
        str: Path to the created folder.
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join("training_runs", now)
    os.makedirs(folder, exist_ok=True)
    return folder

def save_training_info(folder: str) -> None:
    """
    Save training configuration and hyperparameters into a markdown file.
    
    Args:
        folder (str): The folder where the file will be saved.
    """
    info_text = f"""# Training Information

Run Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Environment Config:
{CONFIG}

AI Config:
{AI_CONFIG}

State Dimension: {STATE_SIZE}
Action Space: MultiDiscrete([hand_size={CONFIG.get("hand_size",6)}, {COORD_RES}, {COORD_RES}, 2])
"""
    with open(os.path.join(folder, "training_info.md"), "w") as f:
        f.write(info_text)

def train() -> None:
    """
    Train the Saboteur RL agent using MaskablePPO on a vectorized environment.
    The vectorized environment is created using make_vec_env to parallelize training.
    Evaluation and checkpoint callbacks are used, and training data is logged to TensorBoard.
    """
    run_folder: str = create_run_folder()
    save_training_info(run_folder)
    
    # Number of parallel environments (set in AI_CONFIG; default to 8 if not provided)
    n_envs: int = AI_CONFIG.get("n_envs", 16)
    
    # Create vectorized environment: each env is wrapped with ActionMasker.
    def make_env() -> gym.Env:
        env = SaboteurDiscreteWrapper()
        env = ActionMasker(env, mask_func)
        return env
    
    vec_env = make_vec_env(make_env, n_envs=n_envs)
    
    model: MaskablePPO = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        batch_size=AI_CONFIG.get("batch_size", 1024),
        tensorboard_log=os.path.join(run_folder, "tensorboard"),
    )
    
    # Create evaluation environment (vectorized with 1 env) for evaluation callback.
    eval_env = make_vec_env(make_env, n_envs=3)
    eval_callback: MaskableEvalCallback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=run_folder,
        log_path=run_folder,
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )
    checkpoint_callback: CheckpointCallback = CheckpointCallback(
        save_freq=50000,
        save_path=run_folder,
        name_prefix="checkpoint"
    )
    total_timesteps: int = AI_CONFIG.get("timesteps", 100_000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    model.save(os.path.join(run_folder, "trained_model"))
    vec_env.close()

if __name__ == "__main__":
    train()
