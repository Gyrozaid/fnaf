import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# John Branch

# Constants
DEFAULT_N_TIMESTEPS = 535
DEFAULT_BATTERY = 100
ROOM_NAMES = ["Stage", "Hallway", "Office Entry", "ATTACK"]
N_ROOMS = len(ROOM_NAMES)
ANIMATRONICS = ["Chica"]


@dataclass
class AnimState:
    name: str
    location: int = 0  # index of room name
    move_timer: int = 0  # when hits 0, animatronic can move
    move_period: int = 5  # moving period
    alive: bool = True  # has MDP terminated


class FNAFEnv(gym.Env):
    """Five Nights at Freddy's inspired Gymnasium environment.
    
    The agent must survive by managing battery resources and door controls
    to prevent animatronics from attacking.
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    # Action constants
    TOGGLE_LEFT_DOOR = 0
    TOGGLE_RIGHT_DOOR = 1
    CHECK_CAMERA = 2
    NOOP = 3
    
    def __init__(
        self,
        max_timesteps: int = DEFAULT_N_TIMESTEPS,
        level: int = 1,
        transition_version: int = 2,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.max_timesteps = max_timesteps
        self.level = level
        self.transition_version = transition_version
        self.render_mode = render_mode
        
        # Battery consumption map
        self.battery_consumption_map = {
            0: 1,  # no door/camera
            1: 2,  # one of door or camera
            2: 3,  # two (door + camera OR two doors)
            3: 4,  # two doors + camera
        }
        
        # Mapping which animatronic attacks which door
        self.attack_door_map = {"Bonnie": "left", "Chica": "right"}
        
        # Simplified action space: just 4 discrete actions (ignore camera room for simplicity)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space
        obs_dim = (
            2 +  # timestep, battery
            2 +  # door states
            1 +  # camera active
            len(ANIMATRONICS) * 3  # per anim: location, timer, alive
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state variables
        self.timestep = 0
        self.battery = DEFAULT_BATTERY
        self.left_door_closed = False
        self.right_door_closed = False
        self.camera_focus = None
        self.anims: Dict[str, AnimState] = {}
        self.np_random = None
        
    def _init_move_period(self, anim_name: str) -> int:
        """Get movement period for an animatronic."""
        if anim_name in ("Bonnie", "Chica"):
            return 5
        return 5
    
    def _init_move_timer(self, anim_name: str) -> int:
        """Initialize movement timer with some jitter."""
        period = self._init_move_period(anim_name)
        return self.np_random.integers(1, period + 1)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Handle options
        if options:
            if 'level' in options:
                self.level = options['level']
            if 'transition_version' in options:
                self.transition_version = options['transition_version']
        
        # Reset state
        self.timestep = 0
        self.battery = DEFAULT_BATTERY
        self.left_door_closed = False
        self.right_door_closed = False
        self.camera_focus = None
        
        # Reset animatronics
        self.anims = {}
        for anim_name in ANIMATRONICS:
            self.anims[anim_name] = AnimState(
                name=anim_name,
                location=0,
                move_period=self._init_move_period(anim_name),
                move_timer=self._init_move_timer(anim_name),
                alive=True
            )
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _battery_cost_for_action(
        self,
        will_use_left: bool,
        will_use_right: bool,
        will_use_camera: bool
    ) -> int:
        """Calculate battery cost based on resource usage."""
        count = int(will_use_left) + int(will_use_right) + int(will_use_camera)
        count = max(0, min(3, count))
        return self.battery_consumption_map[count]
    
    def _attempt_anim_move(self, anim: AnimState) -> None:
        """Attempt to move an animatronic if timer is zero."""
        if anim.move_timer > 0:
            return
        
        # Determine probability of advancing
        if self.transition_version == 1:
            p_adv = 0.5
        else:
            # Pr(Advance to next room) = LEVEL / 20
            p_adv = float(self.level) / 20.0
            p_adv = max(0.0, min(1.0, p_adv))
        
        if self.np_random.random() < p_adv:
            # Advance toward Office Entry
            office_idx = ROOM_NAMES.index("Office Entry")
            if anim.location < office_idx:
                anim.location += 1
        
        # Reset timer
        anim.move_timer = anim.move_period
    
    def _decrement_timers(self):
        """Decrement all animatronic movement timers."""
        for anim in self.anims.values():
            if anim.move_timer > 0:
                anim.move_timer -= 1
    
    def _check_death(self) -> Tuple[bool, Optional[str]]:
        """Check if any animatronic kills the player."""
        office_idx = ROOM_NAMES.index("Office Entry")
        
        for name, anim in self.anims.items():
            if not anim.alive:
                continue
            if anim.location >= office_idx:
                side = self.attack_door_map.get(name, "left")
                door_closed = (
                    self.left_door_closed if side == "left" 
                    else self.right_door_closed
                )
                if not door_closed:
                    anim.alive = False
                    return True, name
        
        return False, None
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep within the environment."""
        # Determine resource usage this timestep
        using_left = self.left_door_closed
        using_right = self.right_door_closed
        using_camera = False
        
        # Execute action
        if action == self.TOGGLE_LEFT_DOOR:
            self.left_door_closed = not self.left_door_closed
            using_left = self.left_door_closed
        elif action == self.TOGGLE_RIGHT_DOOR:
            self.right_door_closed = not self.right_door_closed
            using_right = self.right_door_closed
        elif action == self.CHECK_CAMERA:
            # Randomly pick a room to look at
            self.camera_focus = self.np_random.integers(0, N_ROOMS)
            using_camera = True
        elif action == self.NOOP:
            pass
        
        # If battery is zero, all tools become inoperable
        if self.battery <= 0:
            self.left_door_closed = False
            self.right_door_closed = False
            self.camera_focus = None
            using_left = using_right = using_camera = False
        
        # Consume battery
        cost = self._battery_cost_for_action(using_left, using_right, using_camera)
        self.battery = max(0, self.battery - cost)
        
        # Progress animatronic timers and movements
        self._decrement_timers()
        for anim in self.anims.values():
            if anim.move_timer == 0 and anim.alive:
                self._attempt_anim_move(anim)
        
        # Check for death
        dead, killer = self._check_death()
        self.timestep += 1
        
        # Calculate reward and termination
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        if dead:
            reward = -100.0  # Penalty for dying
            terminated = True
            info['killed_by'] = killer
        else:
            # Base survival reward
            reward = 1.0
            
            # Bonus for maintaining good battery
            if self.battery > 50:
                reward += 0.5
            elif self.battery < 20:
                reward -= 0.5  # Penalty for low battery
            
            # Big bonus for completing the night
            if self.timestep >= self.max_timesteps:
                reward += 100.0
                truncated = True
                info['reason'] = 'survived_full_night'
        
        obs = self._get_obs()
        info.update(self._get_info())
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation as a numpy array."""
        obs = []
        
        # Timestep and battery (normalized)
        obs.append(self.timestep / float(self.max_timesteps))
        obs.append(self.battery / float(DEFAULT_BATTERY))
        
        # Door states
        obs.append(1.0 if self.left_door_closed else 0.0)
        obs.append(1.0 if self.right_door_closed else 0.0)
        
        # Camera state
        obs.append(1.0 if self.camera_focus is not None else 0.0)
        
        # Animatronic states
        for anim_name in ANIMATRONICS:
            anim = self.anims[anim_name]
            obs.append(anim.location / float(max(1, N_ROOMS - 1)))
            obs.append(anim.move_timer / float(anim.move_period))
            obs.append(1.0 if anim.alive else 0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information."""
        return {
            'timestep': self.timestep,
            'battery': self.battery,
            'left_door_closed': self.left_door_closed,
            'right_door_closed': self.right_door_closed,
            'camera_focus': self.camera_focus,
            'animatronic_locations': {
                name: anim.location for name, anim in self.anims.items()
            }
        }
    
    def render(self):
        """Render the environment state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
    
    def _render_ansi(self) -> str:
        """Create ASCII representation of the environment."""
        lines = []
        lines.append(f"=== FNAF Environment (t={self.timestep}/{self.max_timesteps}) ===")
        lines.append(f"Battery: {self.battery}%")
        lines.append(f"Left Door: {'CLOSED' if self.left_door_closed else 'OPEN'}")
        lines.append(f"Right Door: {'CLOSED' if self.right_door_closed else 'OPEN'}")
        lines.append(f"Camera: {'ON' if self.camera_focus is not None else 'OFF'}")
        lines.append("\nAnimatronics:")
        for name, anim in self.anims.items():
            status = "ALIVE" if anim.alive else "INACTIVE"
            lines.append(f"  {name}: {ROOM_NAMES[anim.location]} (timer={anim.move_timer}) [{status}]")
        return "\n".join(lines)


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


def evaluate_agent(env: FNAFEnv, agent: DQNAgent, num_episodes: int = 10):
    """Evaluate the trained agent."""
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
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
        
        print(f"Eval Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
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
    print("Starting training...")
    episode_rewards, episode_lengths, losses = train_dqn(
        env, agent, num_episodes=2000, print_freq=100
    )
    
    # Save trained model
    agent.save("fnaf_dqn_model.pt")
    print("\nModel saved to fnaf_dqn_model.pt")
    
    # Evaluate trained agent
    print("\nEvaluating trained agent...")
    evaluate_agent(env, agent, num_episodes=10)
    
    env.close()