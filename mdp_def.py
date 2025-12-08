from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# MDP Constants
DEFAULT_N_TIMESTEPS = 535
DEFAULT_BATTERY = 100
ROOM_NAMES = ["Stage", "Hallway", "Office Entry", "ATTACK"]
N_ROOMS = len(ROOM_NAMES)
ANIMATRONICS = ["Chica", "Freddy"]

# Random Seeds for evaluation, allows for direct comparison between evaluation results
TEST_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

@dataclass
class AnimState:
    name: str
    location: int = 0  # index of the animatronic's room name
    in_office: bool = True  # has the animatronic breached the player's defense?
    focused: bool = False # is the animatronic being focused by the camera?

class FNAFEnv(gym.Env):
    # Action constants
    TOGGLE_LEFT_DOOR = 0
    TOGGLE_RIGHT_DOOR = 1
    CHECK_CAMERA_CHICA = 2
    CHECK_CAMERA_FREDDY = 3
    NOOP = 4
    
    def __init__(self, max_timesteps = DEFAULT_N_TIMESTEPS, render_mode = None):
        super().__init__()
        
        self.max_timesteps = max_timesteps
        self.render_mode = render_mode
        
        # Battery consumption map 
        # key = energy consumption level
        # value = battery used
        self.battery_consumption_map = {
            0: 1,  # no door/camera
            1: 2,  # one of door or camera
            2: 3,  # two (door + camera OR two doors)
            3: 4,  # two doors + camera
        }
        
        # Mapping which animatronic attacks which door
        self.attack_door_map = {"Freddy": "left", "Chica": "right"}
        
        # Simplified action space: NOOP + Toggling each door + Focusing camera on each animatronic
        self.action_space = spaces.Discrete(3 + len(ANIMATRONICS))
        
        # Define observation space
        obs_dim = (
            2 +  # timestep + battery
            2 +  # door states
            len(ANIMATRONICS) * 3  # per anim: location, in_office, focused
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state variables
        self.timestep = 0
        self.battery = DEFAULT_BATTERY
        self.left_door_closed = False
        self.right_door_closed = False
        self.anims: Dict[str, AnimState] = {}
        self.np_random = None
    
    def reset(self, seed = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state
        self.timestep = 0
        self.battery = DEFAULT_BATTERY
        self.left_door_closed = False
        self.right_door_closed = False
        
        # Reset animatronics
        self.anims = {}
        for anim_name in ANIMATRONICS:
            self.anims[anim_name] = AnimState(name=anim_name, location=0, in_office=False, focused=False)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _battery_cost_for_action(self, will_use_left: bool, will_use_right: bool, will_use_camera: bool):
        """Calculate battery cost based on resource usage."""
        count = int(will_use_left) + int(will_use_right) + int(will_use_camera)
        return self.battery_consumption_map[count]
    
    def _attempt_anim_move(self, anim: AnimState):
        """Attempt to move an animatronic."""
        # If the animatronic is focused by the camera, no movement will occur.
        if anim.focused:
            return
        
        # Determine probability of advancing
        p_adv = 0.5 * (1/5)
        
        # Check if advancement occurs or not
        if self.np_random.random() < p_adv:
            # Advance toward ATTACK room
            attack_idx = ROOM_NAMES.index("ATTACK")
            if anim.location < attack_idx:
                anim.location += 1
    
    def _check_death(self):
        """Check if any animatronic attacks the player."""
        attack_idx = ROOM_NAMES.index("ATTACK")
        
        for anim_name, anim_state in self.anims.items():
            # If animatronic is already in office, ignore it
            if anim_state.in_office:
                continue
            if anim_state.location >= attack_idx:
                side = self.attack_door_map.get(anim_name, "left")
                door_closed = (
                    self.left_door_closed if side == "left" 
                    else self.right_door_closed
                )
                # If the door isn't closed when they try to attack, they enter the office and succesfully attack the player
                if not door_closed:
                    anim_state.in_office = True
                    return True, anim_name
                # If the door is closed when they try to attack, reset the animatronic's location
                else:
                    # Reset animatronic location if it tries to attack and the door is closed
                    # Chica goes all the way back to the start (stage)
                    if anim_name == "Chica":
                        anim_state.location = 0
                    # Freddy goes just one space back
                    else:
                        anim_state.location = min(anim_state.location - 1, 0)

                    return False, None
        
        return False, None
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one timestep within the environment."""

        # Determine resource usage this timestep
        using_left = self.left_door_closed
        using_right = self.right_door_closed
        using_camera = False
        for anim_name, anim_state in self.anims.items():
            if anim_state.focused:
                using_camera = True
                break
        
        # Execute action

        # Door usage
        if action == self.TOGGLE_LEFT_DOOR:
            self.left_door_closed = not self.left_door_closed
            using_left = self.left_door_closed
        elif action == self.TOGGLE_RIGHT_DOOR:
            self.right_door_closed = not self.right_door_closed
            using_right = self.right_door_closed
        # Camera usage
        elif self.battery > 0:
            if action == self.CHECK_CAMERA_CHICA:
                # Causes Chica to move back a space.
                curr_location_idx = self.anims["Chica"].location
                self.anims["Chica"].location = max(0, curr_location_idx - 1)
                self.anims["Chica"].focused = True
                using_camera = True
            if action == self.CHECK_CAMERA_FREDDY:
                # Causes Freddy to freeze in place.
                self.anims["Freddy"].focused = True
                using_camera = True
        elif action == self.NOOP:
            pass
        
        # If battery is zero, all tools become inoperable
        if self.battery <= 0:
            self.left_door_closed = False
            self.right_door_closed = False
            for anim_name, anim_state in self.anims.items():
                anim_state.focused = False
            using_left = using_right = using_camera = False
        
        # Consume battery
        cost = self._battery_cost_for_action(using_left, using_right, using_camera)
        self.battery = max(0, self.battery - cost)
        
        # Progress animatronic movements
        for anim in self.anims.values():
            if not anim.in_office:
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
            reward = -1.0  # Penalty for dying
            terminated = True
            info['killed_by'] = killer
        else:
            # Base survival reward
            reward = 1.0
        
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
        
        # Animatronic states
        for anim_name in ANIMATRONICS:
            anim = self.anims[anim_name]
            obs.append(anim.location / float(max(1, N_ROOMS - 1)))
            obs.append(0.0 if anim.in_office else 1.0)
            obs.append(1.0 if anim.focused else 0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self):
        """Get MDP information."""
        return {
            'timestep': self.timestep,
            'battery': self.battery,
            'left_door_closed': self.left_door_closed,
            'right_door_closed': self.right_door_closed,
            'animatronic_locations': {
                name: anim.location for name, anim in self.anims.items()
            },
            'animatronic_focused': {
                name: anim.focused for name, anim in self.anims.items()
            }
        }
    
    def render(self):
        """Render the environment state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
    
    def _render_ansi(self):
        """Create string representation of the environment."""
        lines = []
        lines.append(f"=== FNAF Environment (t={self.timestep}/{self.max_timesteps}) ===")
        lines.append(f"Battery: {self.battery}%")
        lines.append(f"Left Door: {'CLOSED' if self.left_door_closed else 'OPEN'}")
        lines.append(f"Right Door: {'CLOSED' if self.right_door_closed else 'OPEN'}")
        lines.append("\nAnimatronics:")
        for name, anim in self.anims.items():
            lines.append(f"  {name}: {ROOM_NAMES[anim.location]}")
            camera_status = "YES" if anim.focused else "NO"
            lines.append(f"  {"Camera focused?"}: {camera_status}")
        return "\n".join(lines) + "\n"