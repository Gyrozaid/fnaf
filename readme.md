# 🐻 Five Nights at Freddy's - Reinforcement Learning Implementation

A custom Markov Decision Process (MDP) implementation of the survival horror game Five Nights at Freddy's (FNAF), designed for reinforcement learning research and experimentation. 

For a more detailed overview, please refer to our [Final Report](https://github.com/Gyrozaid/fnaf/blob/main/687%20Final%20Report%20(1).pdf).

## 📋 Overview

This project reconstructs FNAF as an MDP environment and applies three existing reinforcement learning algorithms to train RL agents to play the game. The implementation simplifies the original game mechanics while preserving the core strategic challenge: surviving against animatronics by using defensive tools with limited battery power.

## 🎮 Game Mechanics

### Original Game Concept

In Five Nights at Freddy's, players work the night shift at a pizzeria, monitoring animatronics through security cameras from their office. The player must stay in the office and survive until 6 AM without being attacked by any of the four animatronics, each with distinct stochastic movement patterns.

**Player Tools:**
- **Security Cameras**: View 11 different rooms to track animatronic locations
- **Office Doors**: Block animatronics from entering the office
- **Door Lights**: Check for animatronics just outside the office
- **Battery Power**: Limited resource which depletes at a faster rate the more tools you use (i.e. cameras, doors, lights)

### MDP Simplifications

<table>
<tr>
<td width="60%">

To model FNAF as an MDP, we made the following modifications:
1. **Constant Animatronic Location Awareness**: The agent always knows animatronic locations, eliminating the need for exploration with the cameras
2. **Repurposed Camera System**: Cameras have been repurposed to freeze selected animatronics rather than revealing locations
3. **Removed Door Lights**: No longer needed since the agent always knows where the animatronics are
4. **Discrete Time Steps**: Each timestep in the simulation represents one second of in-game time (night duration: 535 timesteps). At each timestep, the agent may choose actions such as focusing their camera on an animatronic, toggling the left/right doors open/closed or turning off the camera to conserve battery. This is meant to roughly model how often a human player would be able to perform actions in-game
5. **Two Animatronics**: Reduced from four to two animatronics with distinct behaviors:
   - **Chica**: Resets to the beginning (stage) when blocked by a door
   - **Freddy**: Resets back one space only if blocked by a door, encouraging the player to focus their camera on him to maximize survival time instead of using the door
6. **Simplified Map**: The map now has a linear layout, with a few rooms leading to the office in a straight line (__Figure 1__). The animatronics also now move in a linear manner, with a 0.1 probability of advancing to the next room at every timestep.

The player begins every episode with full battery, both doors and the camera are inactive and all of the animatronics are initialized at the first room (Stage).

</td>
<td width="40%">

<img width="100%" alt="687 FNAF" src="https://github.com/user-attachments/assets/fe2c74a2-6be9-410d-b154-fee28af0d04b" />

**Figure 1: Our Simplified Map**
</td>
</tr>
</table>

## 🏗️ Project Structure

```
fnaf/
├── mdp_def.py                # MDP environment definition
├── run_algs.py               # Main script to train and evaluate RL algorithms
├── hyperparameter_tuning.py  # Hyperparameter optimization with Optuna
├── simple_heuristic.py       # Evaluates baseline heuristic agent
├── figs/                     # Generated figures and visualizations
├── hp_results/               # Hyperparameter tuning results
├── logs/                     # Recorded training episode data
├── models/                   # Trained models for default and hyperparameter-tuned models
├── tensorboard_logs/         # Logs for Tensorboard visualization
└── 687 Final Report (1).pdf  # Detailed project report
```

## 🚀 Installation

### Prerequisites

Install the required packages in order:

1. **Gymnasium** (OpenAI Gym successor):
```bash
pip install gymnasium
```

2. **Stable Baselines 3** (includes PyTorch):
```bash
pip install stable-baselines3[extra]
```

3. **Optuna** (for hyperparameter tuning):
```bash
pip install optuna
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Gyrozaid/fnaf.git
cd fnaf

# Install dependencies
pip install gymnasium stable-baselines3[extra] optuna

# Run the training and evaluation for all algorithms
python run_algs.py
```
## 🤖 Reinforcement Learning Algorithms

This project implements and compares a baseline heuristic and three RL methods:

1. **Baseline Hueristic** Our proposed "best strategy" of focusing the camera on Freddy and blocking Chica with the door.
2. **Advantage Actor Critic (A2C)**: Popular variant of Actor-Critic model
3. **Deep Q-Network (DQN)**: Neural network-based Q-learning
4. **Proximal Policy Optimization (PPO)**: State-of-the-art policy gradient method

## 🎮 Usage Guide: `run_algs.py`

This script runs training and evaluation of our reinforcement learning agents on the FNAF environment.

### Environment Setup

The script creates separate Monitor environments for each algorithm to track training metrics independently:

```python
dqn_env = Monitor(FNAFEnv(), filename="logs/dqn_training.csv")
a2c_env = Monitor(FNAFEnv(), filename="logs/a2c_training.csv")
ppo_env = Monitor(FNAFEnv(), filename="logs/ppo_training.csv")
```

**Note**: Training logs are saved to CSV files in the `logs/` directory. New training runs will overwrite existing log files.

### Training Modes

#### Tuned Hyperparameters (Recommended)

Train models using optimized hyperparameters discovered through Optuna tuning:

```python
train_dqn_tuned(dqn_env, TRAINING_TIMESTEPS)
train_a2c_tuned(a2c_env, TRAINING_TIMESTEPS)
train_ppo_tuned(ppo_env, TRAINING_TIMESTEPS)
```

#### Default Hyperparameters

Train all three models with their default Stable Baselines3 hyperparameters:

```python
train_default(TRAINING_TIMESTEPS)
```

**Training Options:**
- Comment out any model you don't want to train
- Models are only saved after training completes successfully
- Training timesteps can be adjusted via the `TRAINING_TIMESTEPS` constant

### Evaluation

After training, evaluate all models for comprehensive comparison:

```python
compare_all_methods()
```

**Requirements:**
- All six models must be trained (DQN, A2C, PPO for both tuned and default configurations)
- Evaluation runs agents on a fixed set of random seeds for consistency
- Outputs average return and episode length for each algorithm + a baseline heuristic

### Monitoring Training Progress

#### TensorBoard Visualization

View real-time training metrics and compare algorithm performance:

```bash
tensorboard --logdir ./tensorboard_logs/
```

Then open your browser to the provided URL in the terminal.

**Available Metrics:**
- Episode rewards over time
- Episode lengths
- Exploration rate (DQN)

#### Training Logs

CSV logs are saved in the `logs/` directory:
- `dqn_training.csv` - DQN training episodes
- `a2c_training.csv` - A2C training episodes  
- `ppo_training.csv` - PPO training episodes

Each log contains episode-by-episode statistics including rewards, lengths, and timestamps.

### Typical Workflow

1. **Train models** with tuned hyperparameters:
   ```python
   python run_algs.py  # Trains all models with tuned parameters
   ```

2. **Monitor training** in real-time (optional):
   ```bash
   tensorboard --logdir ./tensorboard_logs/
   ```

3. **Evaluate performance** (runs automatically after training):
   - Compares all trained models + baseline heuristic
   - Outputs performance metrics
   - Generates comparison visualizations (saved to /figs dir)
   
## 📊 Results
<img width="800" alt="sb3_comparison" src="https://github.com/user-attachments/assets/31983d38-a233-4b59-ae03-541b25a20769" />

### Boxplots of reward of each algorithm on an evaluation set of 10 test seeds.

Every algorithm, except for A2C Default, learned the simplified MDP version of FNaF approximately just as well as our heuristic. This resulted in a score of surviving for >50 seconds. In the actual game, power decreases at a rate 10-times slower, so we believe this is a near-optimal score given the decreased battery capacity. DQN and DQN-tuned were able to learn the best with less variance than PPO and A2C. These figures also show that hyperparameter tuning was effective, as it contributed to a large improvement in performance for the A2C agent.

