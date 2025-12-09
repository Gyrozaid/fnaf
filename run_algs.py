import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from mdp_def import FNAFEnv, TEST_SEEDS

# Automatically selects the best device for training
DEVICE = "auto"

def train_dqn(env, total_timesteps):
    """Training DQN agent."""
    print("")
    print("Training DQN")
    print("")
    
    model = DQN(
        "MlpPolicy",
        learning_rate = 0.00020311910371188509,
        gamma = 0.9608865177479774,
        net_arch = [256, 256],
        batch_size = 32,
        buffer_size = 100000,
        target_update_interval = 258,
        exploration_fraction = 0.2680255936799418,
        exploration_final_eps = 0.12320837707847854
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=DEVICE
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save
    model.save("models/fnaf_dqn")
    print()
    print("Model saved to models/fnaf_dqn.zip")
    
    return model


def train_a2c(env, total_timesteps):
    """Training A2C agent."""
    print("")
    print("Training A2C")
    print("")

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate = 0.0003898034895288209,
        gamma = 0.9544658648241585,
        net_arch = [128, 128],
        n_steps = 50,
        ent_coef = 0.00014417404067605557,
        vf_coef = 0.7164978529477865,
        max_grad_norm = 0.5874150956628301,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=DEVICE 
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save
    model.save("models/fnaf_a2c")
    print()
    print("Model saved to models/fnaf_a2c.zip")
    
    return model


def train_ppo(env, total_timesteps):
    """Training PPO agent."""
    print()
    print("Training PPO")
    print()
    
    model = PPO(
        "MlpPolicy",
        learning_rate = 0.0038842777547031426,
        gamma = 0.9811025765286951,
        net_arch = [128, 128],
        n_steps = 2048,
        batch_size = 256,
        n_epochs = 13,
        clip_range = 0.25419343599091215,
        ent_coef = 0.0009444574254983562
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device=DEVICE
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save
    model.save("models/fnaf_ppo")
    print()
    print("Model saved to models/fnaf_ppo.zip")
    
    return model


def evaluate_on_test_seeds(model, model_name, render_mode=None):
    """Evaluate model on fixed test seeds to ensure consistency."""
    print()
    print("Evaluating: ", model_name)
    print()
    
    episode_rewards = []
    episode_lengths = []
    
    for i, seed in enumerate(TEST_SEEDS):
        # Initialize enviornment
        env = FNAFEnv(render_mode=render_mode)
        obs, _ = env.reset(seed=seed)
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        # Run episode
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        # Collect rewards and length
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  Episode {i + 1} (seed={seed}): Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Compute averages and print mean performance stats for the model
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
    # Heuristic, as defined in simple_heuristic.py, serves as a baseline 
    # for a general strategy which we believe will provide very good performance.
    print()
    print("Evaluating: Heuristic")
    print()
    
    # For heuristic, collect same mean performance metrics that we calculate for all other agents.
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
    print(f"  Average Length: {np.mean(heuristic_lengths):.2f} ± {np.std(heuristic_lengths):.2f}")
    
    # 2. Load and evaluate trained models
    
    # Change RENDER_MODE to "human" to view step by step what each agent is doing during evaluation
    # i.e. opening doors, using camera, etc.
    # Also what is happening in the enviornment i.e. where each animatronic is, and if they are being focused by the camera or not, etc.
    RENDER_MODE = None
    dqn_model = DQN.load("models/fnaf_dqn")
    results.append(evaluate_on_test_seeds(dqn_model, "DQN", RENDER_MODE))

    a2c_model = A2C.load("models/fnaf_a2c")
    results.append(evaluate_on_test_seeds(a2c_model, "A2C", RENDER_MODE))
    
    ppo_model = PPO.load("models/fnaf_ppo")
    results.append(evaluate_on_test_seeds(ppo_model, "PPO", RENDER_MODE))
    
    # Print comparison
    print()
    print("COMPARISON SUMMARY")
    print()
    
    # Sort the results based on average reward, in descending order 
    results.sort(key=lambda x: x['avg_reward'], reverse=True)
    
    print(f"\n{'Method':<20} {'Average Reward':<15}")
    print("-" * 35)
    for result in results:
        print(f"{result['name']:<20} {result['avg_reward']:>6.2f} ± {result['std_reward']:<5.2f}")
    
    # Plot comparison boxplot
    plot_comparison(results)


def plot_comparison(results):
    """Create boxplot for the runs of each agent + heuristic on the evaluation test seeds set."""
    names = [r['name'] for r in results]
    rewards = [r['rewards'] for r in results]

    plt.figure()
    bp = plt.boxplot(rewards, tick_labels=names, patch_artist=True)

    for box, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']):
        box.set_facecolor(color)

    plt.title("Performance Comparison Across Test Seeds")
    plt.ylabel("Episode Reward")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("figs/sb3_comparison.png")



if __name__ == "__main__":
    TRAINING_TIMESTEPS = 500000
    
    # Create environments for each env, so they save their logs to different files.
    # Logs override old CSV logs even if the training isnt completed.
    dqn_env = Monitor(FNAFEnv(), filename="logs/dqn_training.csv")
    a2c_env = Monitor(FNAFEnv(), filename="logs/a2c_training.csv")
    ppo_env = Monitor(FNAFEnv(), filename="logs/ppo_training.csv")

    # TRAINING
    # Comment out any model you don't wish to train, training will only overwrite the currently saved model AFTER completion.

    train_dqn(dqn_env, TRAINING_TIMESTEPS)
    train_a2c(a2c_env, TRAINING_TIMESTEPS)
    train_ppo(ppo_env, TRAINING_TIMESTEPS)
    
    # EVALUATION
    # Runs all of the trained agents on a small list of random seeds to ensure consistency.
    # The average return and episode length is returned for each algorithm.
    
    compare_all_methods()
    
    # VIEW METRICS
    # View all runs of the model, stored within the tensorboard_logs directory
    # run in new terminal to view live training / training figs:

    # tensorboard --logdir ./tensorboard_logs/