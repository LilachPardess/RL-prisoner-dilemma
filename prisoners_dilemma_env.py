"""
Prisoner's Dilemma Environment for Gymnasium

This module implements a Prisoner's Dilemma game environment that inherits from
gymnasium.Env.
"""

import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

# Define Actions: C=0, D=1 (Cooperate, Defect)
# Using 0 and 1 is convenient for indexing and calculations.
COOPERATE = 0
DEFECT = 1
ACTIONS = [COOPERATE, DEFECT] 
ACTION_MAP = {COOPERATE: "C", DEFECT: "D"}

# Define the Payoff Matrix (Standard Prisoner's Dilemma)
# Matrix[Your_Action, Opponent_Action] = Your_Payoff, Opponent_Payoff
# Indices: [Row=Your Action, Col=Opponent Action]
# C=0, D=1
#          Opponent -> C (0)      D (1)
# Your_Action
# C (0):     (R=3, R=3)   (S=0, T=5)
# D (1):     (T=5, S=0)   (P=1, P=1)
PAYOFF_MATRIX = np.array([
    [(3, 3), (0, 5)], # Row for Your Action = Cooperate (0)
    [(5, 0), (1, 1)]  # Row for Your Action = Defect (1)
])

class OpponentStrategies:
    """Class containing the logic for the four required opponent strategies."""

    # 1. ALL-C: Always Cooperates
    @staticmethod
    def all_c(history: list) -> int:
        return COOPERATE

    # 2. ALL-D: Always Defects
    @staticmethod
    def all_d(history: list) -> int:
        return DEFECT

    # 3. Tit-for-Tat (TFT): Copies the agent's previous move. Starts with C.
    @staticmethod
    def tit_for_tat(history: list) -> int:
        # history: [(agent_move_t-1, opp_move_t-1), ...]
        # Agent's previous move is the first element of the last pair in history.
        if not history:
            # Game start: Assumed previous state was (C, C).
            return COOPERATE
        
        # history[-1][0] is the agent's move in the last round.
        return history[-1][0]

    # 4. Imperfect Tit-for-Tat: Stochastic. 90% copy, 10% opposite. Starts with C.
    @staticmethod
    def imperfect_tit_for_tat(history: list) -> int:
        # Same initial move as TFT
        if not history:
            return COOPERATE

        agent_prev_move = history[-1][0]
        
        # 90% chance to copy
        if random.random() < 0.9:
            return agent_prev_move
        else:
            # 10% chance to "slip" (do the opposite)
            return 1 - agent_prev_move # 1-0=1 (D), 1-1=0 (C)

class IteratedPrisonersDilemma(gym.Env):
    """
    A Gymnasium environment for the Iterated Prisoner's Dilemma.

    Configurable with different opponent strategies and observation schemes.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Map strategy names to their implementation function
    STRATEGY_MAP = {
        "ALL-C": OpponentStrategies.all_c,
        "ALL-D": OpponentStrategies.all_d,
        "TFT": OpponentStrategies.tit_for_tat,
        "IMPERFECT-TFT": OpponentStrategies.imperfect_tit_for_tat,
    }

    def __init__(self, opponent_strategy: str, memory_scheme: int = 1):
        super().__init__()

        # --- Configuration ---
        if opponent_strategy not in self.STRATEGY_MAP:
            raise ValueError(f"Invalid strategy: {opponent_strategy}")
        self.opponent_strategy = self.STRATEGY_MAP[opponent_strategy]
        
        if memory_scheme not in [1, 2]:
            raise ValueError("Memory scheme must be 1 or 2.")
        self.memory_scheme = memory_scheme

        # --- Gym Setup ---
        # Action Space: {0: Cooperate, 1: Defect}
        self.action_space = spaces.Discrete(2)

        # Observation Space: Depends on memory_scheme (Memory-1 or Memory-2)
        if self.memory_scheme == 1:
            # Memory-1: Previous outcome (Agent's move, Opponent's move) -> (0, 1, 2, 3)
            # C-C=0, C-D=1, D-C=2, D-D=3
            self.observation_space = spaces.Discrete(4)
        else: # memory_scheme == 2
            # Memory-2: (Agent's move t-1, Opponent's move t-1, 
            #           Agent's move t-2, Opponent's move t-2)
            # A vector of 4 moves, each 0 or 1. Total 2^4 = 16 states.
            # We will represent this as a Discrete(16) space after encoding.
            self.observation_space = spaces.Discrete(16)
        
        # Stores the full history of moves (Agent, Opponent)
        # e.g., [(C, C), (D, C), ...]
        self.history = [] 
        self.current_state = None # The state returned to the agent
        self.current_turn = 0

        pass

    def reset(self, seed: Optional[int] = None) -> Tuple[int, dict]:
        """
        Resets the environment for a new episode.
        
        Initial State Handling: Assume both agents Cooperated prior to starting.
        """
        super().reset(seed=seed)
        
        # Clear all history and reset turn counter
        self.history = []
        self.current_turn = 0

        # The initial state is derived from the assumed (C, C) history.
        # This is handled correctly by _get_obs_from_history when self.history is empty.
        observation = self._get_obs_from_history()
        self.current_state = observation
        
        # Return observation and info dictionary
        info = {}
        return observation, info

    def _get_obs_from_history(self) -> int:
            """
            Converts the internal history into the state (observation) 
            according to the configured memory scheme.
            """
            # A. Memory-1: See the outcome of the previous round (Agent's move t-1, Opponent's move t-1)
            if self.memory_scheme == 1:
                # history[-1] is (agent_move_t-1, opp_move_t-1)
                # Encode (A, O) into a single integer: A*2 + O
                # (0, 0) -> 0 | (0, 1) -> 1 | (1, 0) -> 2 | (1, 1) -> 3
                if not self.history:
                    # Start State: (C, C) -> (0, 0) -> 0
                    return 0
                
                agent_prev, opp_prev = self.history[-1]
                return agent_prev * 2 + opp_prev

            # B. Memory-2: See last two moves (A_t-1, O_t-1, A_t-2, O_t-2)
            else: # memory_scheme == 2
                # The state vector has 4 components: 
                # [A_t-1, O_t-1, A_t-2, O_t-2]
                
                # Start State: (C, C, C, C) -> (0, 0, 0, 0)
                # If the history has fewer than 2 rounds, we pad with (C, C) = (0, 0)
                
                # Round t-1
                if len(self.history) >= 1:
                    A_t1, O_t1 = self.history[-1]
                else:
                    A_t1, O_t1 = COOPERATE, COOPERATE # Padding (C, C)

                # Round t-2
                if len(self.history) >= 2:
                    A_t2, O_t2 = self.history[-2]
                else:
                    A_t2, O_t2 = COOPERATE, COOPERATE # Padding (C, C)
                
                # State vector: [A_t1, O_t1, A_t2, O_t2]
                state_vector = [A_t1, O_t1, A_t2, O_t2]
                
                # Binary-to-decimal encoding (MSB is A_t1)
                # This maps the 16 possible vectors to integers 0-15
                # e.g., (1, 0, 0, 0) -> 8 | (0, 0, 0, 0) -> 0 | (1, 1, 1, 1) -> 15
                state_int = 0
                for i, move in enumerate(state_vector):
                    state_int += move * (2**(len(state_vector) - 1 - i))
                
                return state_int