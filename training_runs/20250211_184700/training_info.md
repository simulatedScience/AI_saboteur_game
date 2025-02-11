# Training Information

Run Date: 2025-02-11 18:47:00

Environment Config:
{'num_players': 2, 'hand_size': 6, 'deck': 'src/assets/deck_standard.json', 'AI_TYPES': ['human', 'human']}

AI Config:
{'learning_rate': 0.003, 'gamma': 0.99, 'timesteps': 20000000, 'batch_size': 512, 'n_envs': 24, 'device': 'cpu', 'n_eval_episodes': 3, 'n_steps': 2400, 'log_interval': 1, 'dist_reward_scale': 0.0, 'final_reward_winner': 10}

State Dimension: 2332
Action Space: MultiDiscrete([hand_size=6, 41, 41, 2])
