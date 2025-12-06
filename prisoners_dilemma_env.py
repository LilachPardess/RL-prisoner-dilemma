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

#Part I: Build the Environment (Implementation)
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

    """
        Evaluates a policy by computing the state-value function.
        
        Uses iterative policy evaluation algorithm:
        1. Initialize V(s) = 0 for all states
        2. Repeat until convergence:
           - For each state s:
             - V_new(s) = Σ_a π(a|s) * Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
           - Check convergence: max|V_new(s) - V_old(s)| < theta
        3. Return converged value function V
        
        Args:
            policy: A policy matrix of shape (num_states, num_actions) representing
                   the probability of taking each action in each state.
            gamma: Discount factor for future rewards.
            theta: Convergence threshold for value function updates.
        
        Returns:
            Value function array of shape (num_states,) representing the expected
            return for each state under the given policy.
    """
    def policy_evaluation(self, policy: np.ndarray, gamma: float = 0.9, theta: float = 1e-6) -> np.ndarray:
        # Step 1: Get the number of states and actions
        num_states = self.observation_space.n
        num_actions = self.action_space.n
        
        # Validate policy shape
        if policy.shape != (num_states, num_actions):
            raise ValueError(f"Policy shape {policy.shape} does not match expected ({num_states}, {num_actions})")
        
        # Step 2: Initialize value function V(s) = 0 for all states
        V = np.zeros(num_states)
        
        # Step 3: Iterate until convergence
        while True:
            # Create a new value function for this iteration
            V_new = np.zeros(num_states)
            
            # For each state s in the state space
            for s in range(num_states):
                state_value = 0.0
                
                # For each action a in the action space
                for a in range(num_actions):
                    # Get the policy probability: π(a|s)
                    policy_prob = policy[s, a]
                    
                    # Sum over all possible next states s'
                    action_value = 0.0
                    for next_state in range(num_states):
                        # Get transition probability: P(s'|s,a)
                        transition_prob = self._get_transition_probability(s, a, next_state)
                        
                        # Get immediate reward: R(s,a,s')
                        reward = self._get_reward(s, a, next_state)
                        
                        # Bellman equation component: R(s,a,s') + γ * V(s')
                        bellman_component = reward + gamma * V[next_state]
                        
                        # Weight by transition probability
                        action_value += transition_prob * bellman_component
                    
                    # Weight by policy probability
                    state_value += policy_prob * action_value
                
                # Update value for state s
                V_new[s] = state_value
            
            # Step 4: Check for convergence
            # Compute the maximum change across all states
            delta = np.max(np.abs(V_new - V))
            
            # If converged (change is less than threshold), break
            if delta < theta:
                break
            
            # Update value function for next iteration
            V = V_new.copy()
        
        # Step 5: Return the converged value function
        return V

    def policy_improvement(self, value_function: np.ndarray, gamma: float = 0.9) -> np.ndarray:
        """
        Improves a policy by making it greedy with respect to the value function.
        
        Args:
            value_function: A value function array of shape (num_states,) representing
                          the expected return for each state.
            gamma: Discount factor for future rewards.
        
        Returns:
            Improved policy matrix of shape (num_states, num_actions) representing
            the probability of taking each action in each state.
        """
        pass

    def policy_iteration(self, gamma: float = 0.9, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs policy iteration to find the optimal policy.
        
        Iteratively evaluates and improves the policy until convergence.
        
        Args:
            gamma: Discount factor for future rewards.
            theta: Convergence threshold for value function updates.
        
        Returns:
            A tuple containing:
            - Optimal policy matrix of shape (num_states, num_actions)
            - Optimal value function array of shape (num_states,)
        """
        pass

    def _get_transition_probability(self, current_state: int, action: int, next_state: int) -> float:
        """
        Gets the transition probability P(next_state | state, action).
        
        The transition probability depends on the opponent's strategy:
        - For deterministic opponents (ALL-C, ALL-D, TFT): probability is 1.0 or 0.0
        - For stochastic opponents (IMPERFECT-TFT): probability is between 0.0 and 1.0
        
        Args:
            state: Current state
            action: Action taken by the agent
            next_state: Next state
        
        Returns:
            Transition probability P(s'|s,a)
        """
        # Get opponent action probabilities given current state and agent action
        opp_probs = self._get_opponent_action_probabilities(current_state)
        
        # Initialize total probability
        total_prob = 0.0
        
        # For each possible opponent action
        for opp_action in range(2):  # COOPERATE=0, DEFECT=1
            # Compute what the next state would be if opponent takes this action
            computed_next_state = self._compute_next_state(current_state, action, opp_action)
            
            # If this matches the desired next_state, add the probability
            if computed_next_state == next_state:
                total_prob += opp_probs[opp_action]
        
        return total_prob   

    #TODO : mak this clerer
    #TODO understand the hidtory part
    def _get_opponent_action_probabilities(self, current_state: int) -> np.ndarray:
        """
        Computes the probability distribution over opponent actions given
        the current state and agent's action.
        
        Args:
            state: Current state
            agent_action: Agent's action (COOPERATE or DEFECT)
            
        Returns:
            Array of shape (2,) with probabilities [P(opp_cooperate), P(opp_defect)]
        """
        # Decode state to history
        history = self._decode_state_to_history(current_state)
        
        # Get strategy name
        strategy_name = None
        for name, func in self.STRATEGY_MAP.items():
            if func == self.opponent_strategy:
                strategy_name = name
                break
        
        if strategy_name == "ALL-C":
            return np.array([1.0, 0.0])  # Always cooperate
        
        elif strategy_name == "ALL-D":
            return np.array([0.0, 1.0])  # Always defect
        
        elif strategy_name == "TFT":
            # Deterministic: copies agent's previous move
            if not history:
                # Initial state: starts with C
                return np.array([1.0, 0.0])
            # TFT copies agent's previous move (first element of last history entry)
            agent_prev = history[-1][0]
            if agent_prev == COOPERATE:
                return np.array([1.0, 0.0])
            else:
                return np.array([0.0, 1.0])
        
        elif strategy_name == "IMPERFECT-TFT":
            # Stochastic: 90% copy, 10% opposite
            if not history:
                # Initial state: starts with C
                return np.array([1.0, 0.0])
            
            agent_prev = history[-1][0]
            if agent_prev == COOPERATE:
                # 90% copy (C), 10% opposite (D)
                return np.array([0.9, 0.1])
            else:
                # 90% copy (D), 10% opposite (C)
                return np.array([0.1, 0.9])
        
        # Default: should not reach here
        return np.array([0.5, 0.5])

    def _compute_next_state(self, current_state: int, agent_action: int, opp_action: int) -> int:
        """
        Computes the next state given current state and both actions.
        
        Args:
            current_state: Current state integer
            agent_action: Agent's action
            opp_action: Opponent's action
            
        Returns:
            Next state integer
        """
        if self.memory_scheme == 1:
            # Memory-1: state encodes (A_t-1, O_t-1)
            # Next state encodes (agent_action, opp_action)
            # Encoding: A*2 + O
            return agent_action * 2 + opp_action
        else:
            # Memory-2: current state encodes [A_t-1, O_t-1, A_t-2, O_t-2]
            # Next state encodes [agent_action, opp_action, A_t-1, O_t-1]
            # Decode current state
            A_t1 = (current_state >> 3) & 1  # Most significant bit
            O_t1 = (current_state >> 2) & 1
            # A_t2 and O_t2 are not needed for next state
            
            # Encode next state: [agent_action, opp_action, A_t1, O_t1]
            next_state = (agent_action << 3) + (opp_action << 2) + (A_t1 << 1) + O_t1
            return next_state

    def _get_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Gets the immediate reward for taking action in state and transitioning to next_state.
        
        Args:
            state: Current state
            action: Action taken by the agent
            next_state: Next state after taking the action
        
        Returns:
            Immediate reward value
        """
        pass

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

    def _decode_state_to_history(self, state: int) -> list:
        """
        Decodes a state integer back to a history format that can be used
        by opponent strategies.
        
        Args:
            state: The encoded state integer
            
        Returns:
            A history list in the format [(agent_move, opp_move), ...]
        """
        if self.memory_scheme == 1:
            # Memory-1: state encodes (A_t-1, O_t-1)
            # state = A*2 + O
            # (0, 0) -> 0 | (0, 1) -> 1 | (1, 0) -> 2 | (1, 1) -> 3
            agent_prev = state // 2
            opp_prev = state % 2
            return [(agent_prev, opp_prev)]
        else:
            # Memory-2: state encodes [A_t-1, O_t-1, A_t-2, O_t-2]
            # Decode binary representation
            state_vector = []
            temp = state
            for i in range(4):
                bit = temp % 2
                state_vector.insert(0, bit)
                temp = temp // 2
            
            # state_vector = [A_t1, O_t1, A_t2, O_t2]
            A_t1, O_t1, A_t2, O_t2 = state_vector
            
            # Return history with last two moves (older first, newer last)
            return [(A_t2, O_t2), (A_t1, O_t1)]
