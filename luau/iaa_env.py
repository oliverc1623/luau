# %%
import gymnasium as gym
import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


# %%
class IntrospectiveEnv(MiniGridEnv):
    """Environment from IAA."""

    def __init__(
        self,
        rng: np.random.default_rng,
        size: int = 9,
        agent_start_pos: tuple[int, int] | None = None,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        *,
        locked: bool = False,
        **kwargs: str,
    ):
        self.size = size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.locked = locked
        self.rng = rng
        mission_space = MissionSpace(mission_func=self._gen_mission)

        self.max_steps = max_steps if max_steps is not None else 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            see_through_walls=True,
            max_steps=self.max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        """Generate the mission."""
        return "get to the green goal square"

    def _gen_grid(self, width: int, height: int) -> Grid:
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical and horizontal walls in the center
        for i in range(height):
            self.grid.set(4, i, Wall())
        for i in range(width):
            self.grid.set(i, 4, Wall())

        # Place the goal
        goal_pos = self._get_valid_position(width, height)
        self.put_obj(Goal(), *goal_pos)

        # Place the door(s)
        self.grid.set(4, 2, Door(COLOR_NAMES[0], is_locked=False))
        if self.locked:
            self.grid.set(6, 4, Door(COLOR_NAMES[4], is_locked=True))
        else:
            self.grid.set(6, 4, Door(COLOR_NAMES[0], is_locked=False))
        self.grid.set(4, 6, Door(COLOR_NAMES[0], is_locked=False))

        # Place the key if locked
        if self.locked:
            key_pos = self._get_valid_position(width, height, exclude=[goal_pos])
            self.grid.set(*key_pos, Key(COLOR_NAMES[4]))

        # Place the agent
        agent_pos = self._get_valid_position(width, height, exclude=[goal_pos, key_pos] if self.locked else [goal_pos])
        self.agent_pos = agent_pos
        self.agent_dir = self.rng.integers(0, 4)

        self.mission = "get to the green goal square"

    def _get_valid_position(self, width: int, height: int, exclude=None) -> tuple[int, int]:  # noqa: ANN001
        """Generate a random valid position, avoiding walls and excluded positions."""
        exclude = set(exclude or [])
        while True:
            x = self.rng.integers(1, width - 1)
            y = self.rng.integers(1, height - 1)
            # Skip positions in the central walls and excluded spots
            if (x, y) != (4, 4) and (x != 4 and y != 4) and (x, y) not in exclude:  # noqa: PLR2004
                return (x, y)


class SmallIntrospectiveEnv(MiniGridEnv):
    """Small Introspective Env."""

    def __init__(
        self,
        rng: np.random.default_rng,
        size: int = 6,
        agent_start_pos: tuple[int, int] | None = None,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        *,
        locked: bool = False,
        **kwargs: str,
    ):
        self.rng = rng
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.locked = locked

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "get to the green goal square"

    def _gen_grid(self, width: int, height: int) -> None:
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        vertical_wall = self.rng.integers(2, 4)
        gap = self.rng.integers(1, 5)
        # Generate verical separation wall
        for i in range(height):
            if i != gap:
                self.grid.set(vertical_wall, i, Wall())

        # Place the door and key
        self.grid.set(vertical_wall, gap, Door("yellow", is_locked=self.locked))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        agent_width = self.rng.integers(1, vertical_wall)
        self.agent_pos = (agent_width, gap)
        self.agent_dir = self.rng.integers(0, 4)

        if self.locked:
            # Place the key
            def reject_right_of_door(self, pos: tuple[int, int]) -> bool:  # noqa: ARG001, ANN001
                w, h = pos
                return w > vertical_wall

            self.place_obj(Key("yellow"), reject_fn=reject_right_of_door)

        self.mission = "get to the green goal square"


# %%
def _make_env(seed: int) -> IntrospectiveEnv:
    """Create the environment."""

    def _init() -> IntrospectiveEnv:
        rng = np.random.default_rng(seed)
        env = IntrospectiveEnv(rng=rng, size=9, locked=False, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return _init


def get_env(seed: int, num_envs: int) -> IntrospectiveEnv:
    """Get the environment."""
    envs = [_make_env(seed + i) for i in range(num_envs)]  # Different seed per env
    envs = gym.vector.AsyncVectorEnv(envs, shared_memory=False)
    return envs
