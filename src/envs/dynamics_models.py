import jax.numpy as jnp

from typing import NamedTuple


class LinearModelParams(NamedTuple):
    """Parameters for the linear dynamics model.

    Attributes:
        max_velocity_x: Maximum velocity in x direction (m/s)
        max_velocity_y: Maximum velocity in y direction (m/s)
        max_acceleration_x: Maximum acceleration in x direction (m/s^2)
        max_acceleration_y: Maximum acceleration in y direction (m/s^2)
        friction_coeff: Friction coefficient affecting acceleration (0-1, default 1.0)
    """
    max_velocity_x: float = 20.0
    max_velocity_y: float = 20.0
    max_acceleration_x: float = 5.0
    max_acceleration_y: float = 5.0
    friction_coeff: float = 1.0


class LinearModel:
    """Linear dynamics model for 2D motion.

    Simple double integrator dynamics where the vehicle can accelerate
    independently in x and y directions. No heading angle - the vehicle
    is a point mass that can move in any direction.

    State: [x, y, vx, vy]
    Control: [ax, ay]
    """

    def __init__(self, params: LinearModelParams = LinearModelParams()):
        self.params = params

    @property
    def state_dim(self) -> int:
        return 4

    @property
    def control_dim(self) -> int:
        return 2

    def dynamics(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        """Compute state derivatives."""
        x, y, vx, vy = state
        ax, ay = control

        ax_eff = ax * self.params.friction_coeff
        ay_eff = ay * self.params.friction_coeff

        ax_eff = jnp.clip(ax_eff, -self.params.max_acceleration_x, self.params.max_acceleration_x)
        ay_eff = jnp.clip(ay_eff, -self.params.max_acceleration_y, self.params.max_acceleration_y)

        dx = vx
        dy = vy
        dvx = ax_eff
        dvy = ay_eff

        return jnp.array([dx, dy, dvx, dvy])

    def step(self, state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Integrate dynamics forward one time step using RK4."""
        k1 = self.dynamics(state, control)
        k2 = self.dynamics(state + 0.5 * dt * k1, control)
        k3 = self.dynamics(state + 0.5 * dt * k2, control)
        k4 = self.dynamics(state + dt * k3, control)

        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        next_state = next_state.at[2].set(
            jnp.clip(next_state[2], -self.params.max_velocity_x, self.params.max_velocity_x)
        )
        next_state = next_state.at[3].set(
            jnp.clip(next_state[3], -self.params.max_velocity_y, self.params.max_velocity_y)
        )

        return next_state


class PlanarQuadrotorParams(NamedTuple):
    """Parameters for the planar (2D) quadrotor dynamics model.

    Attributes:
        mass: Quadrotor mass in kg
        inertia: Moment of inertia Ixx in kg*m^2
        arm_length: Distance from center to motor in m
        gravity: Gravitational acceleration in m/s^2
        max_thrust: Maximum total thrust in N
        max_torque: Maximum torque in N*m
    """
    mass: float = 0.18
    inertia: float = 0.00025
    arm_length: float = 0.086
    gravity: float = 9.81
    max_thrust: float = 3.53  # ~2 * 0.18 * 9.81
    max_torque: float = 0.1


class PlanarQuadrotor:
    """Planar (2D) quadrotor dynamics model.

    State: [y, z, phi, y_dot, z_dot, phi_dot] (6D)
    Control: [f, tau] (2D) — thrust and torque

    Equations of motion:
        m * y_ddot = f * sin(phi)
        m * z_ddot = f * cos(phi) - m * g
        Ixx * phi_ddot = tau
    """

    def __init__(self, params: PlanarQuadrotorParams = PlanarQuadrotorParams()):
        self.params = params

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def control_dim(self) -> int:
        return 2

    def dynamics(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        """Compute state derivatives for planar quadrotor."""
        y, z, phi, y_dot, z_dot, phi_dot = state
        f, tau = control

        f = jnp.clip(f, 0.0, self.params.max_thrust)
        tau = jnp.clip(tau, -self.params.max_torque, self.params.max_torque)

        m = self.params.mass
        I = self.params.inertia
        g = self.params.gravity

        dy = y_dot
        dz = z_dot
        dphi = phi_dot

        y_ddot = (f * jnp.sin(phi)) / m
        z_ddot = (f * jnp.cos(phi)) / m - g
        phi_ddot = tau / I

        return jnp.array([dy, dz, dphi, y_ddot, z_ddot, phi_ddot])

    def step(self, state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Integrate dynamics forward one time step using RK4."""
        k1 = self.dynamics(state, control)
        k2 = self.dynamics(state + 0.5 * dt * k1, control)
        k3 = self.dynamics(state + 0.5 * dt * k2, control)
        k4 = self.dynamics(state + dt * k3, control)

        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return next_state

    def hover_control(self) -> jnp.ndarray:
        """Return the control input required to hover."""
        f_hover = self.params.mass * self.params.gravity
        return jnp.array([f_hover, 0.0])
