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

COOPERATE = 0
DEFECT = 1
ACTIONS = [COOPERATE, DEFECT] 
ACTION_MAP = {COOPERATE: "C", DEFECT: "D"}


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


class PrisonersDilemmaEnv(gym.Env):

    # Map strategy names to their implementation function
    STRATEGY_MAP = {
        "ALL-C": OpponentStrategies.all_c,
        "ALL-D": OpponentStrategies.all_d,
        "TFT": OpponentStrategies.tit_for_tat,
        "IMPERFECT-TFT": OpponentStrategies.imperfect_tit_for_tat,
    }

    def __init__(self, opponent_strategy: str, memory_scheme: int = 1):
        """Initialize the Prisoner's Dilemma environment.
        
        Args:
            opponent_strategy: Strategy for the opponent ("ALL-C", "ALL-D", "TFT", "IMPERFECT-TFT")
            memory_scheme: Memory scheme for observations (to be implemented)
        """
        super().__init__()
        # TODO: Implement initialization
        pass
