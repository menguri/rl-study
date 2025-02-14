# Deep Reinforcement Learning Study (DRL-Study)

Welcome to the **Deep Reinforcement Learning Study Repository**!  
Here, I systematically implement and experiment with fundamental algorithms in **Reinforcement Learning (RL)**, following **David Silver’s RL Course** structure (Chapters 1–10).  

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Project Setup](#project-setup)
3. [Chapters Overview](#chapters-overview)
   - [Chapter_01: Introduction to RL and DP](#chapter_01-introduction-to-rl-and-dp)
   - [Chapter_02: Monte Carlo Methods](#chapter_02-monte-carlo-methods)
   - [Chapter_03: Temporal Difference Learning](#chapter_03-temporal-difference-learning)
   - [Chapter_04: Policy Gradient Methods](#chapter_04-policy-gradient-methods)
   - [Chapter_05: Deep Q-Networks (DQN)](#chapter_05-deep-q-networks-dqn)
   - [Chapter_06: Advanced Policy Gradients (A2C, A3C)](#chapter_06-advanced-policy-gradients-a2c-a3c)
   - [Chapter_07: Actor-Critic & PPO](#chapter_07-actor-critic--ppo)
   - [Chapter_08: Model-Based RL](#chapter_08-model-based-rl)
   - [Chapter_09: Multi-Agent RL](#chapter_09-multi-agent-rl)
   - [Chapter_10: RL in Continuous Action Spaces](#chapter_10-rl-in-continuous-action-spaces)
4. [References & Resources](#references--resources)
5. [Contributing](#contributing)
6. [License](#license)

---


- **`common/`**  
  Holds **shared utilities** like:
  - **Replay buffers**
  - **Environment wrappers** or custom `gym` environments
  - **Plotting functions** for rewards, losses, etc.
  - **Helper code** (seeding, logging, etc.)

---

## Project Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/<your-username>/drl-study.git
   cd drl-study