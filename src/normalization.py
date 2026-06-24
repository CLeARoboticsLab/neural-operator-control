"""
Normalization utilities for input data.

This module provides min-max normalization for SetONet inputs based on
the known physical constraints of the grid environment.
"""

import equinox as eqx
import jax.numpy as jnp
from typing import Optional


class MinMaxNormalizer(eqx.Module):
    """
    Min-max normalization that scales data to [0, 1] range.

    Attributes:
        min_vals: Minimum values for each feature dimension
        max_vals: Maximum values for each feature dimension
    """
    min_vals: jnp.ndarray
    max_vals: jnp.ndarray

    def __init__(self, min_vals: jnp.ndarray, max_vals: jnp.ndarray):
        """
        Initialize normalizer with min and max values.

        Args:
            min_vals: Array of minimum values for each dimension
            max_vals: Array of maximum values for each dimension
        """
        self.min_vals = jnp.asarray(min_vals)
        self.max_vals = jnp.asarray(max_vals)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize input to [0, 1] range.

        Args:
            x: Input array of shape (..., n_features)

        Returns:
            Normalized array in [0, 1] range
        """
        return (x - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)

    def inverse(self, x_norm: jnp.ndarray) -> jnp.ndarray:
        """
        Denormalize from [0, 1] back to original range.

        Args:
            x_norm: Normalized input in [0, 1] range

        Returns:
            Denormalized array in original range
        """
        return x_norm * (self.max_vals - self.min_vals + 1e-8) + self.min_vals


class GridEnvironmentNormalizer(eqx.Module):
    """
    Normalizer for grid environment data based on physical constraints.

    This normalizer is designed for the LQR grid environment with known bounds:
    - Position: [-5, 5] for x and y
    - Velocity: Based on max_velocity parameter
    - Actions: Based on max_acceleration parameter

    Attributes:
        state_normalizer: Normalizer for states (position + velocity)
        action_normalizer: Normalizer for actions (accelerations)
    """
    state_normalizer: MinMaxNormalizer
    action_normalizer: MinMaxNormalizer

    def __init__(
        self,
        position_range: tuple = (-5.0, 5.0),
        max_velocity: float = 15.0,
        max_acceleration: float = 5.0
    ):
        """
        Initialize normalizer for grid environment.

        Args:
            position_range: (min, max) range for x and y positions
            max_velocity: Maximum velocity magnitude in m/s
            max_acceleration: Maximum acceleration magnitude in m/s²
        """
        pos_min, pos_max = position_range

        # State: [x, y, vx, vy]
        state_min = jnp.array([pos_min, pos_min, -max_velocity, -max_velocity])
        state_max = jnp.array([pos_max, pos_max, max_velocity, max_velocity])
        self.state_normalizer = MinMaxNormalizer(state_min, state_max)

        # Action: [ax, ay]
        action_min = jnp.array([-max_acceleration, -max_acceleration])
        action_max = jnp.array([max_acceleration, max_acceleration])
        self.action_normalizer = MinMaxNormalizer(action_min, action_max)

    def normalize_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Normalize state vectors."""
        return self.state_normalizer(states)

    def normalize_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Normalize action vectors."""
        return self.action_normalizer(actions)

    def denormalize_states(self, states_norm: jnp.ndarray) -> jnp.ndarray:
        """Denormalize state vectors."""
        return self.state_normalizer.inverse(states_norm)

    def denormalize_actions(self, actions_norm: jnp.ndarray) -> jnp.ndarray:
        """Denormalize action vectors."""
        return self.action_normalizer.inverse(actions_norm)


def create_normalizer_from_config(config: dict) -> GridEnvironmentNormalizer:
    """
    Create a normalizer from a dataset configuration.

    Args:
        config: Configuration dict containing workspace and dynamics_ranges

    Returns:
        Configured GridEnvironmentNormalizer
    """
    # Extract ranges from config
    x_range = config.get('workspace', {}).get('x_range', [-5.0, 5.0])
    position_range = (x_range[0], x_range[1])

    # Use max values from dynamics ranges
    max_velocity = config.get('dynamics_ranges', {}).get('max_velocity_range', [10.0, 15.0])[1]
    max_acceleration = config.get('dynamics_ranges', {}).get('max_acceleration_range', [3.0, 5.0])[1]

    return GridEnvironmentNormalizer(
        position_range=position_range,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration
    )
