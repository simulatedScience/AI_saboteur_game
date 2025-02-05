# RL Agent Training and Design for Saboteur

## Overview

The Saboteur game features a dynamic board, rich card features (including edge types and internal connections), and complex placement rules. Our RL agent must “see” all of these aspects and output a move that is both valid and strategically sound. This document summarizes the design of the agent’s state and action spaces and explains our choices.

## State Representation

The state must capture two main components: the current board and the current player’s hand.

### 1. Board State

- **Card Position:**  
  Each card on the board is placed at continuous (x, y) coordinates.  
- **Edge Types:**  
  Each card has four edges (top, right, bottom, left). Each edge can be one of three types:
  - `wall`
  - `path` (and, for connectivity purposes, we treat `dead-end` similar to `path` but note it separately)
  - `dead-end`
  
  For each edge, we use a one-hot encoding (length 3) to indicate its type.
- **Connection Information:**  
  A card’s internal connections are given as pairs among the four edges. There are 6 possible connection pairs; we encode each as a binary value (1 if connected, 0 otherwise).
- **Special Flags:**  
  - **Hidden Goal Flag:** A binary flag (1 if the card is a goal card that is hidden; 0 otherwise). Once uncovered, a goal card is treated as a path card.
  - **Start Flag:** A binary flag indicating if the card is the start card.

In total, each board card is encoded with:
- 2 continuous values for position  
- 4 edges × 3 one-hot values = 12 binary values  
- 6 binary connection values  
- 2 binary flags  
**Total per card:** 2 + 12 + 6 + 2 = **22 features**

Since the board is variable in size, we fix a maximum (e.g. 100 cards) and pad with zeros when fewer are present. Thus, the board state is represented as a vector of length `100 × 22 = 2200`.

### 2. Hand State

- The player’s hand has a fixed maximum size (e.g. 6 cards).  
- Each card in the hand is encoded using the same 22 features as above.  
- Thus, the hand state vector has length `6 × 22 = 132`.

### 3. Full State Vector

The full state is the concatenation of the board state and hand state. For our chosen maximums, the state vector length is:
- **2200 (board) + 132 (hand) = 2332 features**

This vector fully represents the game situation (all cards on board with their positions and edge/connectivity features, plus the hand).

## Action Space Representation

A valid Saboteur move consists of:
1. **Card Selection:**  
   Choose one card from the hand. (Discrete: 0 to 5)
2. **Placement Coordinates:**  
   Instead of choosing a cell from a fixed grid, the agent outputs continuous (x, y) coordinates. These coordinates are bounded (e.g. between –10 and 10) to cover all plausible placements. After the agent outputs (x, y), the environment will “snap” these coordinates to the closest valid placement (as determined by the game’s rules).
3. **Orientation:**  
   Choose the card’s orientation: 0° or 180° (Discrete: 0 or 1)

Thus, the agent’s action is a tuple:
- `card_index` (Discrete, 6 options)
- `x` (Continuous, Box(low=-10, high=10))
- `y` (Continuous, Box(low=-10, high=10))
- `orientation` (Discrete, 2 options)

This hybrid action space is continuous for the placement coordinates (allowing arbitrary positions) and discrete for card selection and orientation. The environment’s valid placements (via `env.get_valid_placements(card)`) are used to “snap” the (x, y) output to a truly valid location.

## Policy Network Design

The RL agent uses a neural network policy that takes the 2332-dimensional state vector as input and produces three outputs (via three separate “heads”):
1. **Card Selection Head:**  
   A categorical distribution (logits) over 6 cards.
2. **Placement Coordinates Head:**  
   Outputs a 2-dimensional continuous value (the mean of a Gaussian). (For simplicity, we use a fixed standard deviation.)
3. **Orientation Head:**  
   A categorical distribution (logits) over 2 orientations.

The network architecture is configurable via a parameter (a tuple of hidden layer sizes), e.g. `(256, 256)`. The agent’s training algorithm (e.g. PPO from Stable-Baselines3) can then optimize this policy.

## Action Masking

Not every theoretical action is valid according to Saboteur rules. The environment can compute, for a given card and orientation, the set of valid placements. The agent’s continuous coordinate output is “snapped” to the nearest valid placement. Additionally, if no valid placements exist for the chosen card and orientation, the agent should be forced to choose another card or skip its turn.

## Summary

- **State Space:**  
  A 2332-dimensional vector encoding:
  - Up to 100 board cards (each with 22 features)
  - 6 hand cards (each with 22 features)
- **Action Space:**  
  A hybrid space consisting of:
  - Discrete card index (6 options)
  - Continuous (x, y) placement coordinates (e.g. in [-10, 10])
  - Discrete orientation (2 options)
- **Policy:**  
  A configurable neural network with three heads (card selection, coordinate output, orientation selection) that maps the state vector to an action.
- **Action Masking and Snapping:**  
  The environment provides valid placements for each card. The agent’s continuous (x, y) output is snapped to the closest valid placement.
  
This design ensures that every valid Saboteur move can be represented and that the agent receives all necessary information (including detailed card edge types and connectivity) to learn effective play.

_End of Documentation_
