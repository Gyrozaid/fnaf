from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# John Branch

# Constants
DEFAULT_N_TIMESTEPS = 535
DEFAULT_BATTERY = 100
ROOM_NAMES = ["Stage", "Hallway", "Office Entry", "ATTACK"]
N_ROOMS = len(ROOM_NAMES)
ANIMATRONICS = ["Chica"]

# Random Seeds for evaluation, allows for direct comparison between evaluation results
TEST_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


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
        transition_version: int = 1,
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
        # period = self._init_move_period(anim_name)
        # return self.np_random.integers(1, period + 1)
        return self._init_move_period(anim_name)
    
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
        count = int(will_use_left) + int(will_use_right)
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
            p_adv = float(self.level) / 20.0
            p_adv = max(0.0, min(1.0, p_adv))
        
        if self.np_random.random() < p_adv:
            # Advance toward ATTACK room
            attack_idx = ROOM_NAMES.index("ATTACK")
            if anim.location < attack_idx:
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
        attack_idx = ROOM_NAMES.index("ATTACK")
        
        for name, anim in self.anims.items():
            if not anim.alive:
                continue
            if anim.location >= attack_idx:
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
            reward = -1.0  # Penalty for dying
            terminated = True
            info['killed_by'] = killer
        else:
            # Base survival reward
            reward = 1.0

            # Big bonus for completing the night
            # if self.timestep >= self.max_timesteps:
            #     reward += 100.0
            #     truncated = True
            #     info['reason'] = 'survived_full_night'
        
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