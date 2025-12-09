from mdp_def import N_ROOMS, ROOM_NAMES, TEST_SEEDS, FNAFEnv
import numpy as np

def run_heuristic(seed: int):
    env = FNAFEnv(render_mode="human")
    obs, info = env.reset(seed=seed)
    
    total_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Simple heuristic policy
        attack_idx = ROOM_NAMES.index("ATTACK")
        
        right_threat = any(
            anim.location >= attack_idx - 1 and name == "Chica"
            for name, anim in env.anims.items()
        )
        left_threat = any(
            anim.location >= attack_idx - 1 and name == "Freddy"
            for name, anim in env.anims.items()
        )
        
        # Action is now a scalar integer, not an array
        if right_threat and not env.right_door_closed:
            action = FNAFEnv.TOGGLE_RIGHT_DOOR
        elif not right_threat and env.right_door_closed:
            action = FNAFEnv.TOGGLE_RIGHT_DOOR
        elif left_threat and not env.left_door_closed:
            action = FNAFEnv.TOGGLE_LEFT_DOOR
        elif not left_threat and env.left_door_closed:
            action = FNAFEnv.TOGGLE_LEFT_DOOR
        # ALWAYS focus freddy, since once he hits the door, he doensn't go back to the stage he just goes back one step to office entry.
        elif not env.anims["Freddy"].focused:
            action = FNAFEnv.CHECK_CAMERA_FREDDY
        else:
            action = FNAFEnv.NOOP
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            # print(f"\nEpisode ended at t={env.timestep}")
            # print(f"Info: {info}")
            break
    
    # print(f"\nTotal reward: {total_reward}")
    env.close()

    return total_reward, env.timestep
# Example usage
if __name__ == "__main__":
    """Evaluate the heuristic"""
    episode_rewards = []
    episode_lengths = []
    
    for i, seed in enumerate(TEST_SEEDS):
        episode_reward, episode_length = run_heuristic(seed)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Eval Episode {i + 1}: Reward = {episode_reward:.2f}, Length = {episode_length:.2f}")
    
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} \nAverage Length: {np.mean(episode_lengths):.2f}")