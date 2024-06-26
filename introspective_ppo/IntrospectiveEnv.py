import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import numpy as np
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class IntrospectiveEnv(MiniGridEnv):
    def __init__(
        self,
        size=9,
        agent_start_pos=None,
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

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
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(0, height):
            self.grid.set(4, i, Wall())

        # Generate horizontal separation wall
        for i in range(0, width):
            self.grid.set(i, 4, Wall())
        
        # Place the door and key
        self.grid.set(4, 2, Door(COLOR_NAMES[0], is_locked=False))
        self.grid.set(6, 4, Door(COLOR_NAMES[0], is_locked=False))
        self.grid.set(4, 6, Door(COLOR_NAMES[0], is_locked=False))

        # Place a goal square in the bottom-right corner
        self.place_obj(Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class IntrospectiveEnvLocked(MiniGridEnv):
    def __init__(
        self,
        size=9,
        agent_start_pos=None,
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

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
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(0, height):
            self.grid.set(4, i, Wall())

        # Generate horizontal separation wall
        for i in range(0, width):
            self.grid.set(i, 4, Wall())

        goal_width = np.random.randint(1, width-1)
        goal_width = goal_width + 1 if goal_width == 4 else goal_width
        goal_height = np.random.randint(height//2+1, height-1)
        goal_height = goal_height + 1 if goal_height == 4 else goal_height
        self.put_obj(Goal(), goal_width, goal_height)
            
        # Place the door
        self.grid.set(4, 2, Door(COLOR_NAMES[0], is_locked=False))
        self.grid.set(6, 4, Door(COLOR_NAMES[4], is_locked=True))
        self.grid.set(4, 6, Door(COLOR_NAMES[0], is_locked=False))

        # Place the key
        key_width = np.random.randint(1, width-1)
        key_width = key_width + 1 if key_width == 4 else key_width
        key_height = np.random.randint(1, height//2)
        self.grid.set(key_width, key_height, Key(COLOR_NAMES[4]))

        # Place the agent        
        agent_width = np.random.randint(1, width-1)
        agent_width = agent_width + 1 if agent_width == 4 else agent_width
        agent_height = np.random.randint(1, height//2)
        while agent_width == key_width and agent_height == key_height:
            agent_width = np.random.randint(1, width-1)
            agent_width = agent_width + 1 if agent_width == 4 else agent_width
            agent_height = np.random.randint(1, height//2)
        self.agent_pos = (agent_width, agent_height)
        self.agent_dir = np.random.randint(0, 4)

        self.mission = "get to the green goal square"


class SmallUnlockedDoorEnv(MiniGridEnv):
    def __init__(
        self,
        size=9,
        agent_start_pos=None,
        agent_start_dir=0,
        max_steps: int | None = None,
        locked = False,
        **kwargs,
    ):
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
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        vertical_wall = np.random.randint(2, 4)
        gap = np.random.randint(1, 5)
        # Generate verical separation wall
        for i in range(0, height):
            if i != gap:
                self.grid.set(vertical_wall, i, Wall())
        
        # Place the door and key
        self.grid.set(vertical_wall, gap, Door("yellow", is_locked=self.locked))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width-2, height-2)

        # Place the agent
        agent_width = np.random.randint(1,vertical_wall)
        self.agent_pos = (agent_width, gap)
        self.agent_dir = np.random.randint(0, 4)

        if self.locked:
            # Place the key
            def reject_right_of_door(self, pos):
                w, h = pos
                return w > vertical_wall

            self.place_obj(Key("yellow"), reject_fn=reject_right_of_door)

        self.mission = "get to the green goal square"


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        self.env = IntrospectiveEnvLocked(render_mode="rgb_array")
        self.env = RGBImgObsWrapper(self.env)

        self.action_space = spaces.Discrete(7)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, 72, 72), dtype=np.uint8)

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        observation = observation["image"]
        return np.transpose(observation, (2,0,1)), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed)
        observation = observation["image"]
        return np.transpose(observation, (2,0,1)), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()