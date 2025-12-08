import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from mdp_def import FNAFEnv, TEST_SEEDS


def train_dqn(env, total_timesteps=500000):
    """Train DQN agent with Stable-Baselines3."""
    print("")
    print("Training DQN")
    print("")
    
    # Use GPU when possible
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print("Using CPU")
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        tau=1.0,  # Hard update every target_update_interval steps
        gamma=0.995,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[512, 512, 512, 512]),
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=device
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save
    model.save("models/fnaf_dqn_sb3")
    print("\nModel saved to models/fnaf_dqn_sb3.zip")
    
    return model


def train_a2c(env, total_timesteps=500000):
    """Train A2C agent with Stable-Baselines3."""
    print("\n" + "="*60)
    print("Training A2C")
    print("="*60)
    
    # Check for MPS (Apple Silicon GPU) availability
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print("Using CPU")
    
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=5,  # Number of steps to collect before update
        gamma=0.99,
        gae_lambda=1.0,  # GAE parameter (1.0 = Monte Carlo)
        ent_coef=0.01,  # Entropy coefficient
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=device  # Use GPU if available
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save
    model.save("models/fnaf_a2c_sb3")
    print("\nModel saved to models/fnaf_a2c_sb3.zip")
    
    return model


def train_ppo(env, total_timesteps=500000):
    """Train PPO agent with Stable-Baselines3."""
    print("\n" + "="*60)
    print("Training PPO")
    print("="*60)
    
    # Check for MPS (Apple Silicon GPU) availability
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print("Using CPU")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=device  # Use GPU if available
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save
    model.save("models/fnaf_ppo_sb3")
    print("\nModel saved to models/fnaf_ppo_sb3.zip")
    
    return model


def evaluate_on_test_seeds(model, env_class, model_name):
    """Evaluate model on fixed test seeds."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    episode_rewards = []
    episode_lengths = []
    
    for i, seed in enumerate(TEST_SEEDS):
        env = env_class(max_timesteps=535, level=3, transition_version=1)
        obs, _ = env.reset(seed=seed)
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Episode {i + 1} (seed={seed}): Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        env.close()
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print(f"\n{model_name} Results:")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Average Length: {avg_length:.2f} ± {std_length:.2f}")
    
    return {
        'name': model_name,
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'avg_reward': avg_reward,
        'std_reward': std_reward
    }


def compare_all_methods():
    """Compare all methods including heuristic."""
    from simple_heuristic import run_heuristic
    
    results = []
    
    # 1. Heuristic
    print("\n" + "="*60)
    print("Evaluating: Heuristic")
    print("="*60)
    
    heuristic_rewards = []
    heuristic_lengths = []
    
    for i, seed in enumerate(TEST_SEEDS):
        reward, length = run_heuristic(seed)
        heuristic_rewards.append(reward)
        heuristic_lengths.append(length)
        print(f"  Episode {i + 1} (seed={seed}): Reward = {reward:.2f}, Length = {length}")
    
    results.append({
        'name': 'Heuristic',
        'rewards': heuristic_rewards,
        'avg_reward': np.mean(heuristic_rewards),
        'std_reward': np.std(heuristic_rewards)
    })
    
    print(f"\nHeuristic Results:")
    print(f"  Average Reward: {np.mean(heuristic_rewards):.2f} ± {np.std(heuristic_rewards):.2f}")
    
    # 2. Load and evaluate trained models
    env_class = FNAFEnv
    
    try:
        dqn_model = DQN.load("models/fnaf_dqn_sb3")
        results.append(evaluate_on_test_seeds(dqn_model, env_class, "DQN (SB3)"))
    except FileNotFoundError:
        print("\nDQN model not found. Skipping.")
    
    try:
        a2c_model = A2C.load("models/fnaf_a2c_sb3")
        results.append(evaluate_on_test_seeds(a2c_model, env_class, "A2C (SB3)"))
    except FileNotFoundError:
        print("\nA2C model not found. Skipping.")
    
    try:
        ppo_model = PPO.load("models/fnaf_ppo_sb3")
        results.append(evaluate_on_test_seeds(ppo_model, env_class, "PPO (SB3)"))
    except FileNotFoundError:
        print("\nPPO model not found. Skipping.")
    
    # Print comparison
    if len(results) > 1:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        results.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        print(f"\n{'Method':<20} {'Avg Reward':<15}")
        print("-" * 35)
        for result in results:
            print(f"{result['name']:<20} {result['avg_reward']:>6.2f} ± {result['std_reward']:<5.2f}")
        
        # Plot comparison
        plot_comparison(results)


def plot_comparison(results):
    """Create comparison visualization."""
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [r['name'] for r in results]
    rewards = [r['rewards'] for r in results]
    
    bp = ax.boxplot(rewards, tick_labels=names, patch_artist=True)
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(results)]):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Performance Comparison Across Test Seeds', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    pathname = "figs/sb3_comparison.png"
    plt.savefig(pathname, dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to, ", pathname)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--eval', action='store_true', help='Evaluate models')
    parser.add_argument('--algo', type=str, default='all', 
                       choices=['dqn', 'a2c', 'ppo', 'all'],
                       help='Which algorithm to train')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps')
    args = parser.parse_args()
    
    # Create environment
    env = Monitor(FNAFEnv(max_timesteps=535, level=3, transition_version=1))
    
    if args.train:
        # if args.algo in ['dqn', 'all']:
        #     train_dqn(env, args.timesteps)
        
        if args.algo in ['a2c', 'all']:
            train_a2c(env, args.timesteps)
        
        if args.algo in ['ppo', 'all']:
            train_ppo(env, args.timesteps)
    
    if args.eval:
        compare_all_methods()
    
    if not args.train and not args.eval:
        print("Usage:")
        print("  Train: python run_algs.py --train [--algo dqn/a2c/ppo/all] [--timesteps 500000]")
        print("  Eval:  python run_algs.py --eval")
        print("  Both:  python run_algs.py --train --eval")

        # RUN in new terminal to view live training:
        # tensorboard --logdir ./tensorboard_logs/