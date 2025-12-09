import os
import time
import argparse
from typing import Dict, Any
import numpy as np
import optuna

import torch
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt


from mdp_def import FNAFEnv, TEST_SEEDS

#EXAMPLE USAGE IN TERMINAL
#python hyperparameter_tuning.py --algo ppo --n-trials 40 --trial-timesteps 10000 --final-timesteps 300000
#use --skip-hparam to skip tuning and just evaluate a tuned model. Implemented so you don't have to rerun tuning if eval fails
#pass in what algo you want to tune, number of trials to tune for, timesteps to tune for, and timesteps for the final evaluation
#keep trial timesteps lower to save time on training

def make_env(seed: int = 0, max_timesteps=535):
    def _init():
        env = FNAFEnv(max_timesteps=max_timesteps)
        env.reset(seed=seed)
        return env
    return _init

def choose_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

#evaluate deterministic policies
def evaluate_model(model, env, n_eval_episodes=5):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    return mean_reward, std_reward

#objective functions
def objective(trial: optuna.Trial, algo: str, trial_timesteps: int, seed: int):

    
    device = choose_device()

    #standard hyperparams for tuning
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    gamma = trial.suggest_uniform("gamma", 0.95, 0.9999)
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = [128, 128]
    elif net_arch_choice == "medium":
        net_arch = [256, 256]
    else:
        net_arch = [512, 512, 256]

    policy_kwargs = dict(net_arch=net_arch)

    model = None
    env = DummyVecEnv([make_env(seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    #create models
    if algo == "dqn":
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
        target_update_interval = trial.suggest_int("target_update_interval", 100, 2000)
        exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.1, 0.5)
        exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0.01, 0.2)

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=lr,
            gamma=gamma,
            batch_size=batch_size,
            buffer_size=buffer_size,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device
        )

    elif algo == "ppo":
        n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        n_epochs = trial.suggest_int("n_epochs", 3, 20)
        clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
        ent_coef = trial.suggest_loguniform("ent_coef", 1e-5, 1e-1)

        #make sure that batch_size divides n_steps * n_envs evenly
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device
        )

    elif algo == "a2c":
        n_steps = trial.suggest_categorical("n_steps", [5, 10, 20, 50])
        ent_coef = trial.suggest_loguniform("ent_coef", 1e-5, 1e-1)
        vf_coef = trial.suggest_uniform("vf_coef", 0.1, 1.0)
        max_grad_norm = trial.suggest_uniform("max_grad_norm", 0.3, 1.0)

        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    #create fnaf env
    eval_env = FNAFEnv(max_timesteps=535)
    eval_env = Monitor(eval_env)

    #train
    total_steps = 0
    eval_interval = max(1000, trial_timesteps // 5)
    while total_steps < trial_timesteps:
        chunk = min(eval_interval, trial_timesteps - total_steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        total_steps += chunk

        #evaluate training
        mean_reward, std_reward = evaluate_model(model, eval_env, n_eval_episodes=3)
        
        #report to optuna for tuning
        trial.report(mean_reward, total_steps)

        #prune away trials that obviously won't be high performers
        if trial.should_prune():
            model.env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()

    #final evaluation
    mean_reward, std_reward = evaluate_model(model, eval_env, n_eval_episodes=5)

    #save trial model (you can comment this out if you want)
    model.save(f"optuna_tmp/{algo}_trial{trial.number}")

    model.env.close()
    eval_env.close()

    return mean_reward


#main hyperparameter tuning runner
def run_study(algo: str, n_trials: int, trial_timesteps: int, seed: int, study_name: str):
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name, load_if_exists=True)

    def _objective(trial):
        return objective(trial, algo=algo, trial_timesteps=trial_timesteps, seed=seed)

    study.optimize(_objective, n_trials=n_trials, gc_after_trial=True, n_jobs=1)

    print("Study finished. Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")

    #save hyperparameters
    os.makedirs("hp_results", exist_ok=True)
    outpath = os.path.join("hp_results", f"best_params_{algo}.json")
    import json
    with open(outpath, "w") as f:
        json.dump({"value": trial.value, "params": trial.params}, f, indent=2)
    print(f"Saved best params to {outpath}")
    return trial.params, study

def plot_study_progress(study, algo: str):
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = []
    current_best = -np.inf
    
    for v in values:
        current_best = max(current_best, v)
        best_so_far.append(current_best)

    plt.figure(figsize=(8, 5))
    plt.plot(best_so_far, marker="o")
    plt.title(f"Hyperparameter Tuning Progress ({algo})")
    plt.xlabel("Trial Number")
    plt.ylabel("Best Mean Reward")
    plt.grid(True)

    outpath = f"hp_results/{algo}_study_progress.jpg"
    plt.savefig(outpath, format="jpg", dpi=150)
    plt.close()


#train model with the best found hyperparams
def train_final_with_params(algo: str, params: Dict[str, Any], total_timesteps: int, model_output_path: str):
    device = choose_device()
    env = Monitor(FNAFEnv(max_timesteps=535))
    if algo == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=params.get("learning_rate", 1e-3),
            gamma=params.get("gamma", 0.995),
            batch_size=params.get("batch_size", 64),
            buffer_size=params.get("buffer_size", 50000),
            target_update_interval=params.get("target_update_interval", 500),
            exploration_fraction=params.get("exploration_fraction", 0.3),
            exploration_final_eps=params.get("exploration_final_eps", 0.05),
            policy_kwargs=dict(net_arch=_net_from_choice(params.get("net_arch"))),
            verbose=1,
            device=device
        )
    elif algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=params.get("learning_rate", 3e-4),
            n_steps=params.get("n_steps", 2048),
            batch_size=params.get("batch_size", 64),
            n_epochs=params.get("n_epochs", 10),
            gamma=params.get("gamma", 0.99),
            clip_range=params.get("clip_range", 0.2),
            ent_coef=params.get("ent_coef", 0.01),
            policy_kwargs=dict(net_arch=_net_from_choice(params.get("net_arch"))),
            verbose=1,
            device=device
        )
    elif algo == "a2c":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=params.get("learning_rate", 3e-4),
            n_steps=params.get("n_steps", 5),
            gamma=params.get("gamma", 0.99),
            ent_coef=params.get("ent_coef", 0.01),
            vf_coef=params.get("vf_coef", 0.5),
            policy_kwargs=dict(net_arch=_net_from_choice(params.get("net_arch"))),
            verbose=1,
            device=device
        )
    else:
        raise ValueError("Unknown algo")

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_output_path)
    print(f"Saved final model to {model_output_path}")
    env.close()
    return model

#translates an optuna network string into an actual stablebaselines architecture with numbers
def _net_from_choice(choice):
    if isinstance(choice, list):
        return choice
    if choice == "small":
        return [128, 128]
    if choice == "medium":
        return [256, 256]
    return [512, 512, 256]

#seeded evaluation, slight changes to function in run_algs to load in model from path. Return only rewards and lengths
def evaluate_on_test_seeds_path(model_path: str, model_name: str):
    #load in saved model
    if model_name == "dqn":
        model = DQN.load(model_path)
    elif model_name == "ppo":
        model = PPO.load(model_path)
    elif model_name == "a2c":
        model = A2C.load(model_path)
    else:
        raise ValueError("Model doesn't exist!!!")

    #seeded evaluation
    episode_rewards = []
    episode_lengths = []
    
    for i, seed in enumerate(TEST_SEEDS):
        env = FNAFEnv(render_mode="human")
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
    print(f"  Average Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Average Length: {avg_length:.2f} +/- {std_length:.2f}")
    return episode_rewards, episode_lengths


#main program
if __name__ == "__main__":
    
    #args
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo", "a2c"], required=True)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--trial-timesteps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-timesteps", type=int, default=300000,)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--skip-hparam", action="store_true")
    args = parser.parse_args()

    #initialize an optuna study
    study_name = args.study_name or f"fnaf_{args.algo}_study"
    model_out = f"models/fnaf_{args.algo}_best"
    os.makedirs("models", exist_ok=True)

    if args.skip_hparam:
        import json
        with open(f"hp_results/best_params_{args.algo}.json", "r") as f:
            data = json.load(f)
        best_params = data["params"]
    else:
        best_params, study = run_study(
            algo=args.algo,
            n_trials=args.n_trials,
            trial_timesteps=args.trial_timesteps,
            seed=args.seed,
            study_name=study_name
        )

        #visualize progress
        plot_study_progress(study, args.algo)

    final_model = train_final_with_params(
        args.algo,
        best_params,
        total_timesteps=args.final_timesteps,
        model_output_path=model_out
    )

    evaluate_on_test_seeds_path(model_out + ".zip", args.algo)

