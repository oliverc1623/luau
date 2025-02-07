# %%
import numpy as np
from minigrid.core import world_object
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Floor, Goal, Key
from minigrid.minigrid_env import MiniGridEnv


# %%
class FourRoomDoorKey(MiniGridEnv):
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
            self.put_obj(Floor("blue"), split_indx, 2)
            self.put_obj(Floor("blue"), split_indx, 6)
            self.put_obj(Door("red", is_locked=True), 6, split_indx)
            self.place_obj(obj=Key("red"), top=(0, 0), size=(8, split_indx))
        else:
            self.grid.vert_wall(split_indx, 0)
            self.grid.horz_wall(0, split_indx)
            self.put_obj(Floor("blue"), split_indx, 2)
            self.put_obj(Floor("blue"), split_indx, 6)
            self.put_obj(Floor("blue"), 6, split_indx)

        self.place_agent(top=(0, 0), size=(8, 4))
        self.place_obj(Goal(), top=(0, 5), size=(8, 4))


class FourRoomDoorKeyLocked(MiniGridEnv):
    """Environment from IAA."""

    def __init__(
        self,
        size: int = 9,
        agent_start_pos: tuple[int, int] | None = None,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        *,
        locked: bool = True,
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
            self.put_obj(Floor("blue"), split_indx, 2)
            self.put_obj(Floor("blue"), split_indx, 6)
            self.put_obj(Door("red", is_locked=True), 6, split_indx)
            self.place_obj(obj=Key("red"), top=(0, 0), size=(8, split_indx))
        else:
            self.grid.vert_wall(split_indx, 0)
            self.grid.horz_wall(0, split_indx)
            self.put_obj(Floor("blue"), split_indx, 2)
            self.put_obj(Floor("blue"), split_indx, 6)
            self.put_obj(Floor("blue"), 6, split_indx)

        self.place_agent(top=(0, 0), size=(8, 4))
        self.place_obj(Goal(), top=(0, 5), size=(8, 4))


class SmallFourRoomDoorKey(MiniGridEnv):
    """Small Introspective Env."""

    def __init__(
        self,
        size: int = 6,
        agent_start_pos: tuple[int, int] | None = None,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        *,
        locked: bool = False,
        **kwargs: str,
    ):
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
        splitIdx = 2  # self._rand_int(2, width - 2)  # noqa: N806
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = 3  # self._rand_int(1, height - 1)  # noqa: N806
        if self.locked:
            self.put_obj(Door("red", is_locked=self.locked), splitIdx, doorIdx)
            # Place a yellow key on the left side
            self.place_obj(obj=Key("red"), top=(0, 0), size=(splitIdx, height))
        else:
            self.put_obj(Floor("blue"), splitIdx, doorIdx)

        self.mission = "use the key to open the door and then get to the goal"


# %%


class SmallFourRoomDoorKeyLocked(MiniGridEnv):
    """Small Introspective Env."""

    def __init__(
        self,
        size: int = 6,
        agent_start_pos: tuple[int, int] | None = None,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        *,
        locked: bool = True,
        **kwargs: str,
    ):
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
        splitIdx = 2  # self._rand_int(2, width - 2)  # noqa: N806
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = 3  # self._rand_int(1, height - 1)  # noqa: N806
        if self.locked:
            self.put_obj(Door("red", is_locked=self.locked), splitIdx, doorIdx)
            # Place a yellow key on the left side
            self.place_obj(obj=Key("red"), top=(0, 0), size=(splitIdx, height))
        else:
            self.put_obj(Floor("blue"), splitIdx, doorIdx)

        self.mission = "use the key to open the door and then get to the goal"


# %%


class MediumFourRoomDoorKeyLocked(MiniGridEnv):
    """Medium Introspective Env."""

    def __init__(
        self,
        size: int = 8,
        agent_start_pos: tuple[int, int] | None = None,
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        *,
        locked: bool = True,
        **kwargs: str,
    ):
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
        splitIdx = 3  # self._rand_int(2, width - 2)  # noqa: N806
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = 2  # self._rand_int(1, height - 1)  # noqa: N806
        if self.locked:
            self.put_obj(Door("red", is_locked=self.locked), splitIdx, doorIdx)
            # Place a yellow key on the left side
            self.place_obj(obj=Key("red"), top=(0, 0), size=(splitIdx, height))
        else:
            self.put_obj(Floor("blue"), splitIdx, doorIdx)

        self.mission = "use the key to open the door and then get to the goal"


# %%


class MultiRoomGrid(MiniGridEnv):
    """Multi-room grid environment."""

    def __init__(self, config: list, start_rooms: list, goal_rooms: list, room_size: int = 3, max_steps: int = 100, **kwargs: str):
        self.num_rows = len(config)
        self.num_cols = len(config[0])
        self.room_size = room_size
        self.start_rooms = start_rooms
        self.goal_rooms = goal_rooms
        self.config = config
        self.max_tries = 100

        self.width = (self.num_cols * room_size) + (self.num_cols + 1)  # Sum of room sizes + 1 space extra for walls.
        self.height = (self.num_rows * room_size) + (self.num_rows + 1)  # Sum of room sizes + 1 space extra for walls.

        # Placeholder mission space does nothing for now, since we don't want to use it.
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            max_steps=max_steps,
            width=self.width,
            height=self.height,
            **kwargs,
        )

    def _sample_room(self, ul: np.array) -> tuple[int, int]:
        try_idx = 0
        while try_idx < self.max_tries:
            loc = (
                np.random.randint(low=ul[0] + 1, high=ul[0] + (self.room_size + 1)),  # noqa: NPY002
                np.random.randint(low=ul[1] + 1, high=ul[1] + (self.room_size + 1)),  # noqa: NPY002
            )

            if self.grid.get(*loc) is None and (self.agent_pos is None or not np.allclose(loc, self.agent_pos)):
                return loc

            try_idx += 1

        raise ("Failed to sample point in room.")  # noqa: B016

    def _construct_room(self, room_config: list, ul: np.array) -> None:
        # Build default walls on all 4 sides
        self.grid.wall_rect(*ul, self.room_size + 2, self.room_size + 2)

        # Examine each wall in the room config
        for direction, wall in zip(("l", "t", "r", "b"), room_config, strict=False):
            # Carve out an opening or door
            if wall in ("o", "d"):
                if direction == "l":
                    opening_idx = (ul[0], ul[1] + (self.room_size + 2) // 2)
                elif direction == "r":
                    opening_idx = (ul[0] + self.room_size + 1, ul[1] + (self.room_size + 2) // 2)
                elif direction == "t":
                    opening_idx = (ul[0] + (self.room_size + 2) // 2, ul[1])
                elif direction == "b":
                    opening_idx = (ul[0] + (self.room_size + 2) // 2, ul[1] + self.room_size + 1)

                if wall == "o":
                    obj_type = Floor()
                else:
                    obj_type = Door("red", is_open=False, is_locked=True)

                self.grid.set(*opening_idx, obj_type)

    def _gen_grid(self, width: int, height: int) -> None:
        """Generate the grid."""
        # Create an empty grid
        self.grid = Grid(width, height)

        self.mission = ""
        ul = [0, 0]
        key_required = False

        for row in self.config:
            for col in row:
                if "d" in col:
                    key_required = True

                self._construct_room(col, ul)
                ul[0] += self.room_size + 1

            ul[0] = 0
            ul[1] += self.room_size + 1

        # Sample agent start location
        rng = np.random.default_rng()
        room_idx = rng.choice(len(self.start_rooms))
        room_ul = (self.room_size + 1) * np.array(self.start_rooms[room_idx][::-1])
        self.agent_pos = self._sample_room(room_ul)
        rng = np.random.default_rng()
        self.agent_dir = rng.integers(low=0, high=4)

        # Place goal
        rng = np.random.default_rng()
        room_idx = np.array(self.goal_rooms[rng.choice(len(self.goal_rooms))][::-1])
        room_ul = ((self.room_size + 1) * room_idx[0], (self.room_size + 1) * room_idx[1])
        self._place_object(room_ul, Goal())

        if key_required:
            # Place key. Can be in any of the start rooms.
            room_idx = rng.choice(len(self.start_rooms))
            room_ul = (self.room_size + 1) * np.array(self.start_rooms[room_idx][::-1])
            self._place_object(room_ul, Key("red"))

    def _place_object(self, ul: np.array, obj: world_object) -> tuple[int, int]:
        """Place an object in the room."""
        loc = self._sample_room(ul)
        self.put_obj(obj, *loc)

        return loc

    @staticmethod
    def _gen_mission() -> str:
        return "get to the green goal square"


# %%


class SimpleEnv(MiniGridEnv):
    """Simple Env."""

    def __init__(
        self,
        size: int = 8,
        agent_start_pos: tuple = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        locked: int = 0,
        **kwargs: str,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2

        self.locked = locked

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=256,
            **kwargs,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "grand mission"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        if self.locked:
            self.put_obj(Goal(), width - 2, height - 6)
        else:
            self.put_obj(Goal(), width - 2, height - 2)
