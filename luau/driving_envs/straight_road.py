# %%


from metadrive.envs import MetaDriveEnv


# %%


class StraightRoad(MetaDriveEnv):
    """A single environment class that can be configured into different variants."""

    VARIANTS = {  # noqa: RUF012
        "S": {
            "map": "S",
            "traffic_density": 0.1,
            "random_lane_num": True,
            "num_scenarios": 10,
        },
        "S_long": {
            "map": "SSS",
            "traffic_density": 0.1,
            "random_lane_num": True,
            "num_scenarios": 10,
        },
        "S_dense": {
            "map": "S",
            "traffic_density": 0.3,
            "random_lane_num": True,
            "num_scenarios": 10,
        },
    }

    # 2. Override the __init__ method.
    def __init__(self, variant: str = "S", config: dict | None = None):
        """Initialize the environment with a specific variant."""
        # Ensure the chosen variant is valid
        if variant not in self.VARIANTS:
            msg = f"Unknown variant: {variant}. Available variants: {list(self.VARIANTS.keys())}"
            raise ValueError(msg)

        # Start with the chosen variant's configuration as the base
        final_config = self.VARIANTS[variant].copy()

        # If the user provided an additional config dict, update our
        # variant config with it. This allows for manual overrides.
        if config:
            final_config.update(config)

        # 3. Call the parent constructor with the final, resolved configuration.
        super().__init__(config=final_config)
        print(f"Initialized '{variant}' variant with map '{self.config['map']}'")


# %%
