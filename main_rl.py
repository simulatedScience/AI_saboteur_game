# main_rl.py
"""
Main entry point for running a multi-agent simulation of the Saboteur game.
This module uses the AI opponent types specified in CONFIG["AI_TYPES"].
If the provided list is shorter than the number of players or includes "human",
the remaining players are treated as human (here, simulated by a dummy skip agent).
At the end of the game, final rewards are printed.
"""


from src.saboteur_env import SaboteurEnv
from src.agents.random_ai import RandomAgent
from src.agents.rule_based_ai import RuleBasedAgent
from src.config import CONFIG, AI_CONFIG

def get_agent(agent_type: str, env: SaboteurEnv):
    """
    Create an agent instance based on the given type.

    Args:
        agent_type (str): The type of agent ("random", "rule-based", or "human").
        env (SaboteurEnv): The game environment.

    Returns:
        An agent instance or None for human players.
    """
    if agent_type == "random":
        return RandomAgent(env)
    elif agent_type == "rule-based":
        return RuleBasedAgent(env)
    else:
        # For human players, in this simulation we simply return None,
        # and later use a dummy action (skip) for human players.
        return None

def main():
    env = SaboteurEnv()
    obs, info = env.reset()
    num_players = env.num_players

    # Read AI opponent types from CONFIG; e.g., ["random", "rule-based"].
    ai_types: list[str] = CONFIG.get("AI_TYPES", [])
    # Pad agent types with "human" if necessary.
    while len(ai_types) < num_players:
        ai_types.append("human")

    agents: list = []
    for agent_type in ai_types:
        agent = get_agent(agent_type, env)
        agents.append(agent)

    # Main simulation loop.
    while not env.done:
        current_player = env.current_player
        agent = agents[current_player]
        if agent is None:
            # For human players, simulate a skip.
            action = (-1, (0, 0), 0)
        else:
            action = agent.act(current_player)
        obs, reward, done, _, info = env.step(action)
    # Game over: print final rewards.
    print("Game over!")
    print("Final rewards:")
    print(info.get("final_rewards", {}))

if __name__ == "__main__":
    main()
