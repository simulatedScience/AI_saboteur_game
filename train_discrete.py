# train_discrete.py
"""
Training script for the Saboteur RL agent using Stable-Baselines3's MaskablePPO.
This version uses a vectorized environment (via make_vec_env) for parallelization,
which should greatly increase training speed (from ~60fps to >10000fps).
TensorBoard logging is enabled and a new run folder is created for each training run.
"""
# standard library imports
import os
from datetime import datetime
import json
# third party imports
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.wrappers import ActionMasker
# local imports
from src.agents.mask_function import mask_func
from src.saboteur_env import SaboteurEnv, STATE_SIZE, COORD_RES
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
    Save training configuration and hyperparameters into a JSON file.
    
    Args:
        folder (str): The folder where the file will be saved.
    """
    training_info = {
        "Run Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Environment Config": CONFIG,
        "AI Config": AI_CONFIG,
        "State Dimension": STATE_SIZE,
        "Action Space": {
            "hand_size": CONFIG["hand_size"],
            "coord_res_1": COORD_RES,
            "coord_res_2": COORD_RES,
            "actions": 2
        }
    }
    
    with open(os.path.join(folder, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=4)

def train() -> None:
    """
    Train the Saboteur RL agent using MaskablePPO on a vectorized environment.
    The vectorized environment is created using make_vec_env to parallelize training.
    Evaluation and checkpoint callbacks are used, and training data is logged to TensorBoard.
    """
    run_folder: str = create_run_folder()
    save_training_info(run_folder)
    
    # Number of parallel environments (set in AI_CONFIG; default to 8 if not provided)
    n_envs: int = AI_CONFIG["n_envs"]
    
    # Create vectorized environment: each env is wrapped with ActionMasker.
    def make_env() -> gym.Env:
        env = SaboteurEnv()
        env = ActionMasker(env, mask_func)
        return env
    
    vec_env = make_vec_env(make_env, n_envs=n_envs)
    
    model: MaskablePPO = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=2,
        batch_size=AI_CONFIG["batch_size"],
        tensorboard_log=os.path.join(run_folder, "tensorboard"),
        device=AI_CONFIG["device"],
        n_steps=AI_CONFIG["n_steps"]
    )
    
    # Create evaluation environment (vectorized with 1 env) for evaluation callback.
    # eval_env = make_vec_env(make_env, n_envs=3)
    # eval_callback: MaskableEvalCallback = MaskableEvalCallback(
    #     eval_env,
    #     best_model_save_path=run_folder,
    #     log_path=run_folder,
    #     eval_freq=10000,
    #     n_eval_episodes=AI_CONFIG["n_eval_episodes"],
    #     deterministic=True,
    #     render=False
    # )
    checkpoint_callback: CheckpointCallback = CheckpointCallback(
        save_freq=10_000,
        save_path=run_folder,
        name_prefix="checkpoints/checkpoint",
    )
    total_timesteps: int = AI_CONFIG["timesteps"]
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=AI_CONFIG["log_interval"],
        callback=[checkpoint_callback]
    )
    model.save(os.path.join(run_folder, "trained_model"))
    vec_env.close()

if __name__ == "__main__":
    import cProfile
    import pstats
    # profile = cProfile.Profile()
    # profile.enable()
    train()
    # profile.disable()
    # stats = pstats.Stats(profile).sort_stats('tottime')
    # stats.print_stats(30)

    # to see training data live in tensorboard run:
    # tensorboard --logdir training_runs