# AI Saboteur - Card Game

This project aims to implement the card game Saboteur and train AI agents to play it, potentially learning deceptive behaviour in  a small-scale experiment.

**Goal:** Investigate under which conditions the AI agents learn deceptive behaviour, what kinds of behaviour emerge?

Different communication methods between agents can be implemented:  
**(a)** (AI) players can ask other players where the gold is, choose to answer (right, wrong, don't know, refuse answer) (using small NNs trained from scratch for this game)  
**(b)** A chat the AI agents can use to communicate. (using LLM agents fine tuned via RL to play this game.)





## Milestones

1. implement path cards, dead ends, path card play, distance to gold measurement.
2. implement basic GUI to test
3. train early agents to play cards to reach gold. ("innocent miner agents")
4. add peek cards and non-equal rewards (first player gets rewarded more, then decreasing rewards in turn order) -> retrain agents with memory (LSTMs?) ("naive miner agents")
5. add rockfall and saboteur roles, adjust rewards based on role -> train new saboteur agent, train new miner agent in env with saboteur
6. build comparison tools -> compare saboteur, miner, naive miner, innocent miner agents' behaviours
7. add tool-related action cards & basic communication -> retrain all agents
8. add chat & LLM agents -> retraining may be difficult due to larger agents -> training on HPC ...?

### non-ordered milestones:
- make player-count adjustable
- add dead-end path cards
- make available cards easily adjustable. -> config file with each card type and number of cards.
- make reward cards easily adjustable -> config table with reward amount, card count for each player count.
- make saboteur/ miner distribution easily adjustable -> config table with miner card count, saboteur card count for each player count.
- decide on observation and action spaces (a: for regular NN agents, b: for LLM agents)

During development reevaluate the next steps based on observations. Adding complexity may not be necessary if very interesting behaviour is observed before.
