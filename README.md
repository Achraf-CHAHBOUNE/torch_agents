# DQN Trading Agent — Deep Reinforcement Learning for Stock Trading

A Deep Q-Network (DQN) reinforcement-learning agent that learns a simple trading policy over
historical price data. At each step the agent observes a window of recent prices and chooses
one of three actions — **sit, buy, or sell** — learning to maximize cumulative profit.

Two implementations are included:
- `torch_agent.py` — **PyTorch** version.
- `Agent.py` — **Keras / TensorFlow** version.

## How it works

- **State**: a normalized window of previous days' prices.
- **Actions**: `0 = sit`, `1 = buy`, `2 = sell`.
- **Learning**: experience replay (`deque` memory) with ε-greedy exploration
  (`epsilon` decays from 1.0 → 0.01) and discounting (`gamma = 0.95`).
- **Models**: trained networks are saved under `models/`; training data lives in `data/`.

## Tech stack

**Python · PyTorch · Keras / TensorFlow · NumPy**

## Getting started

```bash
git clone https://github.com/Achraf-CHAHBOUNE/torch_agents.git
cd torch_agents
pip install torch numpy        # (for the Keras version: keras tensorflow)
# train / evaluate using the agent in torch_agent.py
```

> This is a learning project exploring value-based RL (DQN) on financial time series. It is a
> research/demo implementation, **not** financial advice or a production trading system.

## Status

Functional DQN prototype with both PyTorch and Keras backends. Possible next steps: a clean
training script with CLI args, evaluation plots, and a documented sample dataset.
