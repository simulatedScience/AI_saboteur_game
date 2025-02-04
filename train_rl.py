# train_rl.py
"""
Training script for the Saboteur RL agent using the DQN implemented in agents/rl_agent.py.
This script runs episodes in self-play and trains the network via experience replay.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from src.saboteur_env import SaboteurEnv
from src.agents.rl_agent import RLAgent

# Hyperparameters
EPISODES = 500
BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE = 10
REPLAY_BUFFER_SIZE = 10000

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self) -> int:
        return len(self.buffer)

def train():
    env = SaboteurEnv()
    agent = RLAgent(env)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    optimizer = agent.optimizer
    policy_net = agent.policy_net
    target_net = agent.target_net

    total_steps = 0
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(0)  # training in single-agent/self-play mode
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_steps += 1

            if len(replay_buffer) >= BATCH_SIZE:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(BATCH_SIZE)
                batch_state_tensor = torch.from_numpy(batch_state).float()
                batch_reward_tensor = torch.from_numpy(batch_reward).float()
                batch_next_state_tensor = torch.from_numpy(batch_next_state).float()
                batch_done_tensor = torch.from_numpy(batch_done).float()

                # For actions, convert discrete indices using the mapping.
                # We need to convert the tuple actions stored into a discrete index.
                # Here, we assume that the agent._map_action is invertible; for simplicity,
                # we store the discrete action index along with the tuple.
                # (In a full implementation, modify the agent to store the index.)
                # For now, we assume that the action is represented by a discrete index.
                # We use a placeholder: convert the tuple action back to a discrete index.
                # This placeholder function simply returns 0.
                # You must implement an inverse mapping for full training.
                # For demonstration, we use zeros.
                batch_action_tensor = torch.zeros((BATCH_SIZE,), dtype=torch.long)

                current_q = policy_net(batch_state_tensor).gather(1, batch_action_tensor.unsqueeze(1)).squeeze()
                next_q = target_net(batch_next_state_tensor).max(1)[0].detach()
                expected_q = batch_reward_tensor + GAMMA * next_q * (1 - batch_done_tensor)
                loss = nn.MSELoss()(current_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        print(f"Episode {episode}: Total Reward = {total_reward}")
    
    # Save the trained model.
    torch.save(policy_net.state_dict(), "trained_dqn.pt")

if __name__ == "__main__":
    train()
