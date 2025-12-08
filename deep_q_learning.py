import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mdp_def import TEST_SEEDS, FNAFEnv

# ===== Deep Q-Learning Implementation =====

class DQN(nn.Module):
    """Deep Q-Network with fully connected layers."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Learning agent."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.995,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.998,
        buffer_capacity: int = 50000,
        batch_size: int = 256,
        target_update_freq: int = 10
    ):
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q-networks
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.update_count = 0
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors (keep on CPU initially, then transfer once)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Return loss as Python float to avoid keeping computation graph
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


def train_dqn(
    env: FNAFEnv,
    agent: DQNAgent,
    num_episodes: int = 1000,
    print_freq: int = 50
):
    """Train the DQN agent."""
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Select and perform action
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_length = np.mean(episode_lengths[-print_freq:])
            avg_loss = np.mean(losses[-print_freq:]) if losses[-print_freq:] else 0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Avg Loss: {avg_loss:.4f}")
    
    return episode_rewards, episode_lengths, losses


def evaluate_agent(env: FNAFEnv, agent: DQNAgent):
    """Evaluate the trained agent."""
    episode_rewards = []
    episode_lengths = []
    
    for i, seed in enumerate(TEST_SEEDS):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        episode_length = 0
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = agent.select_action(state, explore=False)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Eval Episode {i + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f}")
    
    return episode_rewards, episode_lengths


# Example usage
if __name__ == "__main__":
    # Create environment
    env = FNAFEnv(max_timesteps=535, level=3, transition_version=1)
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.995,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.998,
        buffer_capacity=50000,
        batch_size=256,
        target_update_freq=10
    )
    
    # Train agent
    # print("Starting training...")
    # episode_rewards, episode_lengths, losses = train_dqn(
    #     env, agent, num_episodes=4000, print_freq=100
    # )
    
    # Save trained model
    # model_path = "models/fnaf_dqn_model.pt"
    # agent.save(model_path)
    # print("\nModel saved to, " + model_path)

    model_path = "models/fnaf_dqn_model.pt"
    agent.load(model_path)
    
    # Evaluate trained agent
    print("\nEvaluating trained agent...")
    evaluate_agent(env, agent)
    
    env.close()