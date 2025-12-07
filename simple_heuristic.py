from mdp_def import N_ROOMS, ROOM_NAMES, TEST_SEEDS, FNAFEnv
import numpy as np

def run_heuristic(seed: int):
    env = FNAFEnv(max_timesteps=535, level=3, transition_version=1)
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
            anim.location >= attack_idx - 1 and name == "Bonnie"
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
        # elif env.np_random.random() < 0.1:
        #     # Randomly check camera 10% of the time
        #     action = FNAFEnv.CHECK_CAMERA
        else:
            action = FNAFEnv.NOOP
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"\nEpisode ended at t={env.timestep}")
            print(f"Info: {info}")
            break
    
    print(f"\nTotal reward: {total_reward}")
    env.close()

    return total_reward
# Example usage
if __name__ == "__main__":
    reward_sum = []
    for seed in TEST_SEEDS:
        total_reward = run_heuristic(seed)
        reward_sum.append(total_reward)

    print("\nHEURISTIC" )
    print("Average Reward over 10 evaluation iterations: ", np.average(reward_sum))