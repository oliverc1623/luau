# %%
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

        mission_space = MissionSpace(mission_func=self._gen_mission)

        self.max_steps = max_steps
        if self.max_steps is None:
            self.max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            # Set this to True for maximum speed
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

        # Generate verical separation wall
        for i in range(height):
            self.grid.set(4, i, Wall())

        # Generate horizontal separation wall
        for i in range(width):
            self.grid.set(i, 4, Wall())

        rng = np.random.default_rng()
        goal_width = rng.integers(1, width - 1)
        goal_width = goal_width + 1 if goal_width == 4 else goal_width  # noqa: PLR2004
        goal_height = rng.integers(height // 2 + 1, height - 1)
        goal_height = goal_height + 1 if goal_height == 4 else goal_height  # noqa: PLR2004
        self.put_obj(Goal(), goal_width, goal_height)

        # Place the door
        self.grid.set(4, 2, Door(COLOR_NAMES[0], is_locked=False))
        if self.locked:
            self.grid.set(6, 4, Door(COLOR_NAMES[4], is_locked=self.locked))
        else:
            self.grid.set(6, 4, Door(COLOR_NAMES[0], is_locked=self.locked))
        self.grid.set(4, 6, Door(COLOR_NAMES[0], is_locked=False))

        # Place the key
        key_width = rng.integers(1, width - 1)
        key_width = key_width + 1 if key_width == 4 else key_width  # noqa: PLR2004
        key_height = rng.integers(1, height // 2)
        self.grid.set(key_width, key_height, Key(COLOR_NAMES[4]))

        # Place the agent
        agent_width = rng.integers(1, width - 1)
        agent_width = agent_width + 1 if agent_width == 4 else agent_width  # noqa: PLR2004
        agent_height = rng.integers(1, height // 2)
        while agent_width == key_width and agent_height == key_height:
            agent_width = rng.integers(1, width - 1)
            agent_width = agent_width + 1 if agent_width == 4 else agent_width  # noqa: PLR2004
            agent_height = rng.integers(1, height // 2)
        self.agent_pos = (agent_width, agent_height)
        self.agent_dir = rng.integers(0, 4)

        self.mission = "get to the green goal square"
