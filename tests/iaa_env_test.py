import gymnasium as gym
import numpy as np
import pytest
from minigrid.core.world_object import Door, Key

from luau.iaa_env import IntrospectiveEnv


@pytest.fixture()
def rng() -> np.random.default_rng:
    """Create a random number generator for testing."""
    return np.random.default_rng(47)


def test_initialization(rng: np.random.default_rng) -> None:
    """Test the initialization of the environment."""
    env = IntrospectiveEnv(rng=rng, size=9, agent_start_pos=(1, 1), agent_start_dir=1, locked=True)
    assert env.size == 9
    assert env.agent_start_pos == (1, 1)
    assert env.agent_start_dir == 1
    assert env.locked is True
    assert env.max_steps == 4 * 9**2


def test_gen_mission() -> None:
    """Test the generation of the mission."""
    mission = IntrospectiveEnv._gen_mission()  # noqa: SLF001
    assert mission == "get to the green goal square"


def test_gen_grid(rng: np.random.default_rng) -> None:
    """Test the generation of the grid."""
    env = IntrospectiveEnv(rng=rng, size=9)
    env.reset()
    assert isinstance(env.grid.get(4, 2), Door)
    assert isinstance(env.grid.get(4, 6), Door)
    assert isinstance(env.grid.get(6, 4), Door)
    assert isinstance(env.grid.get(env.agent_pos[0], env.agent_pos[1]), type(None))
    assert env.grid.get(env.agent_pos[0], env.agent_pos[1]) is None
    assert env.grid.get(env.agent_pos[0], env.agent_pos[1]) is None


def test_locked_env(rng: np.random.default_rng) -> None:
    """Test the generation of the grid in a locked environment."""
    env = IntrospectiveEnv(rng=rng, size=9, locked=True)
    env.reset()
    key_found = False
    for i in range(9):
        for j in range(9):
            if isinstance(env.grid.get(i, j), Key):
                key_found = True
                break
    assert key_found is True


def test_agent_position(rng: np.random.default_rng) -> None:
    """Test the agent position."""
    env = IntrospectiveEnv(rng=rng, size=9)
    env.reset()
    assert 1 <= env.agent_pos[0] < 9
    assert 1 <= env.agent_pos[1] < 9
    assert env.agent_pos != (4, 4)


def test_agent_position_after_reset(rng: np.random.default_rng) -> None:
    """Test the agent position remains the same after environment reset."""
    env = IntrospectiveEnv(rng=rng, size=9)
    env.reset(seed=47)
    initial_agent_pos = env.agent_pos
    initial_agent_dir = env.agent_dir
    env.reset(seed=47)
    assert env.agent_pos != initial_agent_pos
    assert env.agent_dir != initial_agent_dir


def test_async_vector_env(rng: np.random.default_rng) -> None:
    """Test the AsyncVectorEnv with IntrospectiveEnv."""

    def make_env(seed: int) -> IntrospectiveEnv:
        def _init() -> IntrospectiveEnv:
            vec_rng = np.random.default_rng(seed)
            env = IntrospectiveEnv(rng=vec_rng, size=9)
            env.reset(seed=seed)
            return env

        return _init

    seed = 47
    num_envs = 4
    envs = [make_env(seed + i) for i in range(num_envs)]
    async_env = gym.vector.AsyncVectorEnv(envs, shared_memory=False)
    _, _ = async_env.reset()
    assert async_env.env_fns[0]().agent_pos != async_env.env_fns[1]().agent_pos
