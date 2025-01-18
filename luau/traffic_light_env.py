# %%

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class TrafficLightEnv(MiniGridEnv):
    """
    TrafficLight / Wait-Gate environment for MiniGrid.

    --------------------------------------------------
    - A cell in front of the goal behaves like a "gate" that automatically
      opens (green light) and closes (red light) periodically.
    - If the agent is on the gate cell when it closes, or tries to move
      onto a closed gate, the episode ends in failure.
    - The agent must learn to wait until the gate is open before passing.
    """

    def __init__(
        self,
        size: int = 5,
        agent_start_pos: tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        open_duration: int = 3,
        closed_duration: int = 3,
        max_steps: int | None = None,
        **kwargs: dict,
    ):
        """
        Define class cars for the TrafficLightEnv class.

        :param size: Size of the grid.
        :param agent_start_pos: (x, y) starting position of the agent.
        :param agent_start_dir: Starting direction of the agent (0: right, 1: down, etc.).
        :param open_duration: Number of steps the gate stays open in each cycle.
        :param closed_duration: Number of steps the gate stays closed in each cycle.
        :param max_steps: Max steps before episode terminates (if None, defaults to 4 * size^2).
        """
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Gate timing settings
        self.open_duration = open_duration
        self.closed_duration = closed_duration
        self.cycle_length = open_duration + closed_duration

        self.traffic_light = Door("green", is_open=True, is_locked=False)

        if max_steps is None:
            max_steps = 4 * (size**2)

        # Define a simple mission for clarity
        mission_space = MissionSpace(mission_func=lambda: "get to the green goal square")

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate dividing wall
        mid = width // 2
        for i in range(height):
            self.grid.set(mid, i, Wall())

        # Place traffic light (door)
        self.put_obj(self.traffic_light, mid, 1)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Define the gate cell to be just in front of the goal
        # For instance, if goal is at (width-2, height-2), place the gate at (width-2, height-3)
        self.gate_pos = (width - 2, height - 3)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def _is_gate_open(self, step_count: int) -> bool:
        """
        Check if the gate is open given the current step_count.

        The gate cycles with period self.cycle_length: 'open_duration' steps
        open, followed by 'closed_duration' steps closed.
        """
        # We use mod arithmetic to see where we are in the cycle
        t = step_count % self.cycle_length
        return t < self.open_duration

    def step(self, action: int) -> tuple[np.array, float, bool, bool, dict]:
        """
        Set custom step logic to enforce gate constraints.

        - If the gate is closed and the agent is on it (or moves onto it), fail.
        - Otherwise, step as normal.
        """
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        front_cell = self.grid.get(*self.front_pos)

        # Take a step using the parent class
        obs, reward, done, truncated, info = super().step(action)

        # Check if the gate is closed
        gate_is_closed = not self._is_gate_open(self.step_count)
        if gate_is_closed:
            self.traffic_light.color = "red"
        else:
            self.traffic_light.color = "green"

        not_clear = False
        if front_cell is not None:
            not_clear = front_cell.type == "door" and gate_is_closed

        # If the agent is on the gate cell and the gate is closed => fail
        if tuple(self.agent_pos) == self.gate_pos and gate_is_closed:
            done = True
            reward = 0  # or another penalty if you prefer
            info["failure_reason"] = "Agent on gate when closed"

        if action == self.actions.forward and not_clear:
            # Check gate state *prior* to increment
            # (i.e., step_count - 1 was the previous step's time)
            done = True
            reward = -1
            info["failure_reason"] = "Ran onto a closed gate"

        return obs, reward, done, truncated, info
