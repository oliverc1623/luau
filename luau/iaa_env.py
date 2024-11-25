# %%
import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
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
            self.max_steps = 10 * size**2

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
        split_indx = 4

        if self.locked:
            self.grid.vert_wall(split_indx, 0)
            self.grid.horz_wall(0, split_indx)
            self.put_obj(Door("blue", is_locked=False), split_indx, 2)
            self.put_obj(Door("blue", is_locked=False), split_indx, 6)
            self.put_obj(Door("red", is_locked=True), 6, split_indx)
            self.place_obj(obj=Key("red"), top=(0, 0), size=(8, split_indx))
        else:
            self.grid.vert_wall(split_indx, 0, length=2)
            self.grid.vert_wall(split_indx, 3, length=2)
            self.grid.vert_wall(split_indx, 4, length=2)
            self.grid.vert_wall(split_indx, 7, length=2)
            self.grid.horz_wall(4, split_indx, length=2)
            self.grid.horz_wall(0, split_indx, length=6)
            self.grid.horz_wall(7, split_indx, length=2)
        self.place_agent(top=(0, 0), size=(8, 4))
        self.place_obj(Goal(), top=(0, 4), size=(8, 4))


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
            max_steps = 10 * size**2

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

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)  # noqa: N806
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, height - 2)  # noqa: N806
        self.put_obj(Door("yellow", is_locked=self.locked), splitIdx, doorIdx)

        # Place a yellow key on the left side
        if self.locked:
            self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))

        self.mission = "use the key to open the door and then get to the goal"


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
