# train_discrete.py
"""
Training script for the Saboteur RL agent using Stable-Baselines3's MaskablePPO.
The environment is wrapped with an ActionMasker (using our mask_func) to provide valid action masking.
A new folder (named with the current datetime) is created for each training run to save the model,
checkpoints, and a training_info.md file.
"""

import os
from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

from src.agents.mask_function import mask_func
from src.agents.saboteur_discrete_wrapper import SaboteurDiscreteWrapper, STATE_SIZE, COORD_RES
from src.config import AI_CONFIG, CONFIG

def create_run_folder() -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join("training_runs", now)
    os.makedirs(folder, exist_ok=True)
    return folder

def save_training_info(folder: str) -> None:
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

def train():
    run_folder = create_run_folder()
    save_training_info(run_folder)
    env = SaboteurDiscreteWrapper()
    env = ActionMasker(env, mask_func)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(run_folder, "tensorboard"),
        **AI_CONFIG
    )
    eval_env = SaboteurDiscreteWrapper()
    eval_env = ActionMasker(eval_env, mask_func)
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
    total_timesteps = AI_CONFIG.get("timesteps", 100_000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            eval_callback,
            checkpoint_callback
            ]
        )
    model.save(os.path.join(run_folder, "trained_model"))
    env.close()

if __name__ == "__main__":
    train()