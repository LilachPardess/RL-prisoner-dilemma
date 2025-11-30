# RL Prisoner's Dilemma

A reinforcement learning project exploring the Prisoner's Dilemma game using various RL algorithms.

## Overview

This project implements and experiments with reinforcement learning algorithms to solve or study the classic Prisoner's Dilemma game. The Prisoner's Dilemma is a fundamental problem in game theory that demonstrates why two rational individuals might not cooperate, even if it appears that it is in their best interest to do so.

## Project Structure

```
RL-prisoner-dilemma/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── environment.py      # Prisoner's Dilemma game environment
│   ├── agents.py           # RL agent implementations
│   └── training.py         # Training scripts
├── experiments/
│   └── results/            # Training results and plots
└── notebooks/
    └── analysis.ipynb      # Analysis and visualization notebooks
```

## Features

- Implementation of the Prisoner's Dilemma game environment
- Multiple RL algorithm support (Q-learning, DQN, Policy Gradient, etc.)
- Training and evaluation pipelines
- Visualization of agent behavior and learning curves

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RL-prisoner-dilemma
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training an Agent

```bash
python src/training.py --algorithm qlearning --episodes 1000
```

### Running Experiments

```bash
python experiments/run_experiments.py
```

## Algorithms

- **Q-Learning**: Tabular Q-learning for discrete state-action spaces
- **DQN**: Deep Q-Network for function approximation
- **Policy Gradient**: Direct policy optimization methods

## Results

Results and visualizations will be saved in the `experiments/results/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Axelrod, R. (1984). The Evolution of Cooperation

