# %%


from typing import Any

from metadrive.envs import MetaDriveEnv


# %%


class CustomMetaDriveEnv(MetaDriveEnv):
    """Custom MetaDrive Environment."""

    def __init__(self, config: dict):
        super().__init__(config)

    def reset(self, seed: int, options: Any = None) -> tuple:  # noqa: ARG002
        """Call the parent reset method."""
        obs, info = super().reset(seed)
        return obs, info

    def render(self, r: int = 0, t: int = 0):  # noqa: ANN201
        """Call the parent render method."""
        return super().render(
            mode="topdown",
            window=False,
            screen_size=(400, 400),
            camera_position=(100, 7),
            scaling=2,
            screen_record=True,
            text={
                "Timestep": t,
                "Reward": f"{r:0.2f}",
            },
        )


# %%
