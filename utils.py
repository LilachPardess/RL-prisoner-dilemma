import numpy as np


def print_policy(policy, value_function, env, opponent_strategy=None, gamma=None):
    """
    Print policy and value function in a readable format.
    
    Args:
        policy: Policy matrix of shape (num_states, num_actions)
        value_function: Value function array of shape (num_states,)
        env: IteratedPrisonersDilemma environment instance
        opponent_strategy: Optional string to display opponent strategy name
        gamma: Optional discount factor to display
    """
    num_states = env.observation_space.n
    memory_scheme = env.memory_scheme
    
    print("\n" + "=" * 80)
    print("OPTIMAL POLICY AND VALUE FUNCTION")
    print("=" * 80)
    if opponent_strategy:
        print(f"Opponent Strategy: {opponent_strategy}")
    if gamma is not None:
        print(f"Discount Factor (γ): {gamma}")
    print(f"Memory Scheme: {memory_scheme}")
    print()
    
    # Print policy table
    print("POLICY (π):")
    print("-" * 80)
    if memory_scheme == 1:
        # Memory-1: 4 states
        state_names = ["(C, C)", "(C, D)", "(D, C)", "(D, D)"]
        print(f"{'State':<15} {'State ID':<10} {'P(C)':<10} {'P(D)':<10} {'Best Action':<15} {'V(s)':<10}")
        print("-" * 80)
        for s in range(num_states):
            state_name = state_names[s]
            p_cooperate = policy[s, 0]
            p_defect = policy[s, 1]
            best_action_idx = np.argmax(policy[s])
            best_action = "Cooperate (C)" if best_action_idx == 0 else "Defect (D)"
            v_value = value_function[s]
            print(f"{state_name:<15} {s:<10} {p_cooperate:<10.4f} {p_defect:<10.4f} {best_action:<15} {v_value:<10.4f}")
    else:
        # Memory-2: 16 states
        print(f"{'State':<25} {'State ID':<10} {'P(C)':<10} {'P(D)':<10} {'Best Action':<15} {'V(s)':<10}")
        print("-" * 80)
        for s in range(num_states):
            # Decode state: [A_t-1, O_t-1, A_t-2, O_t-2]
            state_vector = []
            temp = s
            for i in range(4):
                bit = temp % 2
                state_vector.insert(0, bit)
                temp = temp // 2
            A_t1, O_t1, A_t2, O_t2 = state_vector
            
            # Format state name
            action_map = {0: "C", 1: "D"}
            state_name = f"({action_map[A_t1]},{action_map[O_t1]})→({action_map[A_t2]},{action_map[O_t2]})"
            
            p_cooperate = policy[s, 0]
            p_defect = policy[s, 1]
            best_action_idx = np.argmax(policy[s])
            best_action = "Cooperate (C)" if best_action_idx == 0 else "Defect (D)"
            v_value = value_function[s]
            print(f"{state_name:<25} {s:<10} {p_cooperate:<10.4f} {p_defect:<10.4f} {best_action:<15} {v_value:<10.4f}")
    
    print()


def print_comparison(comparison):
    print(f"\nRandom vs Best Policy:")
    print(f"  Random: {comparison['random_reward']:.2f}")
    print(f"  Best:   {comparison['policy_reward']:.2f}")
    print(f"  Diff:   {comparison['difference']:.2f}\n")


def print_section_header(title, opponent_strategy=None, memory_scheme=None):
    """Print a section header with optional opponent strategy and memory scheme."""
    print("\n" + "#" * 80)
    print(f"# {title}")
    print("#" * 80)
    if opponent_strategy:
        print(f"# Opponent Strategy: {opponent_strategy}")
    if memory_scheme is not None:
        print(f"# Memory Scheme: {memory_scheme}")
    print("#" * 80)


def print_subsection(title):
    """Print a subsection separator with title."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def print_experiment_end(opponent_strategy=None):
    """Print experiment end marker."""
    end_text = "# END OF EXPERIMENT"
    if opponent_strategy:
        end_text += f" - {opponent_strategy}"
    print("\n" + "#" * 80)
    print(end_text)
    print("#" * 80 + "\n")

