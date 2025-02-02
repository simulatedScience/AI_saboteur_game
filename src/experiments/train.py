# experiments/train.py
"""
This file will eventually contain code to train AI agents on the Saboteur game.
For now, it demonstrates a simple loop interacting with the environment.
"""

from .game_env import SaboteurEnv
from .agents.rule_based import RuleBasedAgent

def train():
    env = SaboteurEnv()
    agent = RuleBasedAgent(env)
    obs, _ = env.reset()

    for step in range(100):  # placeholder training loop
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step}: Reward = {reward}")
        if done or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    train()
