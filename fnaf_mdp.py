import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

#John Branch

#constants
DEFAULT_N_TIMESTEPS = 535
DEFAULT_BATTERY = 100 * 1
ROOM_NAMES = [
    "Stage", "PirateCove", "East1", "East2", "Hallway", "Office_Entry",
    "Room6", "Room7", "Room8", "Room9", "Room10"
]
N_ROOMS = len(ROOM_NAMES)
ANIMATRONICS = ["Bonnie", "Chica"]


#action class
class Actions:
    TOGGLE_LEFT_DOOR = 0
    TOGGLE_RIGHT_DOOR = 1
    CHECK_CAMERA = 2
    NOOP = 3


@dataclass
class AnimState:
    name: str
    location: int = 0 #index of room name
    move_timer: int = 0 #when hits 5, animatronic can move
    move_period: int = 5 #moving period
    alive: bool = True #has MDP terminated


@dataclass
class FnaFMDP:
    max_timesteps: int = DEFAULT_N_TIMESTEPS
    battery: int = DEFAULT_BATTERY
    timestep: int = 0
    left_door_closed: bool = False
    right_door_closed: bool = False
    camera_focus: Optional[int] = None  #none or room index
    anims: Dict[str, AnimState] = field(default_factory=dict)
    rng_seed: Optional[int] = None

    #dynamics
    level: int = 1  # used in transition prob: Pr(advance) = level / 20 (if version=2)
    transition_version: int = 2  # 1 = 50/50, 2 = level-based
    battery_consumption_map = {
        0: 1,  # no door/camera
        1: 2,  # one of door or camera
        2: 3,  # two (door + camera OR two doors)
        3: 4,  # two doors + camera
    }

    #mapping which animatronic attacks which door: default Bonnie->left, Chica->right
    attack_door_map: Dict[str, str] = field(default_factory=lambda: {"Bonnie": "left", "Chica": "right"})

    #episode state
    done: bool = False

    def __post_init__(self):
        self.rng = random.Random(self.rng_seed)
        #initialize animatronics
        for a in ANIMATRONICS:
            #default spawn: Stage = index 0
            self.anims[a] = AnimState(name=a, location=0, move_timer=self._init_move_timer(a),
                                      move_period=self._init_move_period(a), alive=True)

    def _init_move_period(self, anim_name: str) -> int:
        #default movement periods (seconds). Can be adjusted per anim.
        if anim_name in ("Bonnie", "Chica"):
            return 5
        # defaults for others (not used here)
        return 5

    def _init_move_timer(self, anim_name: str) -> int:
        # initial countdown before eligible to move (randomized somewhat)
        period = self._init_move_period(anim_name)
        # jitter initial timer to spread movement
        return self.rng.randint(1, period)

    def reset(self, seed: Optional[int] = None, level: Optional[int] = None, transition_version: Optional[int] = None):
        """Reset the environment to the initial state.

        Args:
            seed: optional RNG seed for reproducibility.
            level: difficulty level (overrides self.level if provided).
            transition_version: 1 or 2 for how transitions are modeled.
        """
        if seed is not None:
            self.rng_seed = seed
            self.rng = random.Random(seed)
        if level is not None:
            self.level = level
        if transition_version is not None:
            self.transition_version = transition_version

        self.timestep = 0
        self.battery = DEFAULT_BATTERY
        self.left_door_closed = False
        self.right_door_closed = False
        self.camera_focus = None
        self.done = False

        # reset anims
        for a in ANIMATRONICS:
            self.anims[a] = AnimState(
                name=a,
                location=0,
                move_period=self._init_move_period(a),
                move_timer=self._init_move_timer(a),
                alive=True
            )

        return self._get_state()

    def _battery_cost_for_action(self, will_use_left: bool, will_use_right: bool, will_use_camera: bool) -> int:
        # compute how many tools are used this timestep (doors considered closed count as 'in use' if closed)
        count = 0
        # treat "using a door" as the door being closed at this timestep (cost counts for each closed door)
        if will_use_left:
            count += 1
        if will_use_right:
            count += 1
        if will_use_camera:
            count += 1
        # clamp to 3 index mapping (0..3)
        count = max(0, min(3, count))
        return self.battery_consumption_map[count]

    def _attempt_anim_move(self, anim: AnimState) -> None:
        """If anim.move_timer == 0, decide whether it advances to next room."""
        if anim.move_timer > 0:
            return
        # determine probability of advancing
        if self.transition_version == 1:
            p_adv = 0.5
        else:
            # specified: Pr(Advance to next room) = LEVEL / 20
            p_adv = float(self.level) / 20.0
            # clamp [0,1]
            p_adv = max(0.0, min(1.0, p_adv))

        roll = self.rng.random()
        if roll < p_adv:
            # advance 1 step toward the Office_Entry (we'll assume office is index 'Office_Entry')
            # which index is Office_Entry?
            try:
                office_idx = ROOM_NAMES.index("Office_Entry")
            except ValueError:
                office_idx = N_ROOMS - 1
            # simple path: increase location by 1 unless already at or beyond office
            if anim.location < office_idx:
                anim.location += 1
        # after attempting move, reset move_timer to move_period
        anim.move_timer = anim.move_period

    def _decrement_timers(self):
        for anim in self.anims.values():
            if anim.move_timer > 0:
                anim.move_timer -= 1

    def _check_death(self) -> Tuple[bool, Optional[str]]:
        """Check if any animatronic kills the player this timestep.

        Returns (dead_flag, killer_name or None).
        """
        # death occurs if an anim is at Office_Entry and the corresponding door is open (unclosed)
        try:
            office_idx = ROOM_NAMES.index("Office_Entry")
        except ValueError:
            return False, None

        for name, anim in self.anims.items():
            if not anim.alive:
                continue
            if anim.location >= office_idx:
                # anim at attack position
                side = self.attack_door_map.get(name, "left")
                door_closed = (self.left_door_closed if side == "left" else self.right_door_closed)
                if not door_closed:
                    # death
                    anim.alive = False
                    return True, name
        return False, None

    def step(self, action: int, camera_room: Optional[int] = None) -> Tuple[Dict, int, bool, Dict]:
        """Perform action for one timestep and progress the environment.

        Args:
            action: integer action (use Actions.*)
            camera_room: if action is CHECK_CAMERA, which room index to focus (0..N_ROOMS-1)

        Returns:
            next_state (dict), reward (int), done (bool), info (dict)
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        # Determine what resources will be used this timestep.
        using_left = False
        using_right = False
        using_camera = False

        # NOTE: in this implementation door toggles are immediate (take effect this timestep).
        # If you want them to take effect in next timestep, modify accordingly.
        if action == Actions.TOGGLE_LEFT_DOOR:
            # toggle door state
            self.left_door_closed = not self.left_door_closed
            using_left = self.left_door_closed  # if now closed, it's being used
        elif action == Actions.TOGGLE_RIGHT_DOOR:
            self.right_door_closed = not self.right_door_closed
            using_right = self.right_door_closed
        elif action == Actions.CHECK_CAMERA:
            # focusing camera consumes power
            # set camera focus to the requested room (if valid)
            if camera_room is None or not (0 <= camera_room < N_ROOMS):
                # if invalid room, treat as noop (no camera cost)
                self.camera_focus = None
                using_camera = False
            else:
                self.camera_focus = camera_room
                using_camera = True
        elif action == Actions.NOOP:
            pass
        else:
            raise ValueError("Unknown action")

        # If battery already zero, tools are inoperable (doors open and camera doesn't work)
        if self.battery <= 0:
            # force doors open and camera off
            self.left_door_closed = False
            self.right_door_closed = False
            self.camera_focus = None
            using_left = using_right = using_camera = False

        # battery consumption this timestep
        cost = self._battery_cost_for_action(using_left, using_right, using_camera)
        self.battery = max(0, self.battery - cost)

        # Progress anim timers and possibly move anims
        self._decrement_timers()
        # After decrement, attempt moves for those with move_timer == 0
        for anim in self.anims.values():
            if anim.move_timer == 0 and anim.alive:
                self._attempt_anim_move(anim)

        # check death after movement and after door toggles take effect
        dead, killer = self._check_death()
        reward = 0
        info = {}
        self.timestep += 1

        if dead:
            # negative reward / terminal
            reward = 0  # alternative: reward= -100 for dying; but you requested +1 per survived second
            self.done = True
            info['killed_by'] = killer
            return self._get_state(), reward, self.done, info

        # Survived this timestep -> reward +1
        reward = 1

        # Terminal if timestep reached max or battery zero and tools inoperable and anim kills later?
        if self.timestep >= self.max_timesteps:
            self.done = True
            info['reason'] = 'survived_full_night'
            return self._get_state(), reward, self.done, info

        # If battery is zero, doors and camera are inoperable but the episode continues until kill or time end.
        # Note: we don't terminate just because battery is zero.

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> Dict:
        """Return a compact serializable state (dict)."""
        state = {
            'timestep': self.timestep,
            'battery': self.battery,
            'left_door_closed': int(self.left_door_closed),
            'right_door_closed': int(self.right_door_closed),
            'camera_focus': self.camera_focus,
            'anims': {
                name: {
                    'location': anim.location,
                    'move_timer': anim.move_timer,
                    'move_period': anim.move_period,
                    'alive': anim.alive
                } for name, anim in self.anims.items()
            },
            'done': self.done
        }
        return state

    def state_to_vector(self) -> List[float]:
        """Return a flat numeric vector representation (for input to an RL agent).
        Order (example): [timestep_norm, battery_norm, left_closed, right_closed, camera_onehot..., anim locations...]
        """
        vec = []
        vec.append(self.timestep / float(self.max_timesteps))
        vec.append(self.battery / float(DEFAULT_BATTERY))
        vec.append(1.0 if self.left_door_closed else 0.0)
        vec.append(1.0 if self.right_door_closed else 0.0)

        # camera one-hot across rooms + 1 for None
        for i in range(N_ROOMS):
            vec.append(1.0 if self.camera_focus == i else 0.0)
        vec.append(1.0 if self.camera_focus is None else 0.0)

        # anims: location normalized and move_timer / move_period
        for a in ANIMATRONICS:
            anim = self.anims[a]
            vec.append(anim.location / float(max(1, N_ROOMS - 1)))
            vec.append(anim.move_timer / float(anim.move_period))
            vec.append(1.0 if anim.alive else 0.0)

        return vec


# --- Example usage (random policy) ---
if __name__ == "__main__":
    env = FnaFMDP(max_timesteps=535, rng_seed=42, level=3, transition_version=2)
    state = env.reset(seed=42)
    total_reward = 0
    for t in range(env.max_timesteps):
        # simple heuristic agent: if anim close to office on right side, close right door; similarly left.
        # find anim locations
        right_threat = any((anim.location >= ROOM_NAMES.index("Office_Entry") - 1 and name == "Chica")
                           for name, anim in env.anims.items())
        left_threat = any((anim.location >= ROOM_NAMES.index("Office_Entry") - 1 and name == "Bonnie")
                          for name, anim in env.anims.items())

        if right_threat and not env.right_door_closed:
            action = Actions.TOGGLE_RIGHT_DOOR
            next_state, r, done, info = env.step(action)
        elif left_threat and not env.left_door_closed:
            action = Actions.TOGGLE_LEFT_DOOR
            next_state, r, done, info = env.step(action)
        else:
            # randomly check a camera room 10% of the time
            if env.rng.random() < 0.1:
                room = env.rng.randrange(N_ROOMS)
                next_state, r, done, info = env.step(Actions.CHECK_CAMERA, camera_room=room)
            else:
                next_state, r, done, info = env.step(Actions.NOOP)

        total_reward += r
        if done:
            print("Episode ended at t =", env.timestep, "info:", info)
            break

    print("Total reward:", total_reward)
