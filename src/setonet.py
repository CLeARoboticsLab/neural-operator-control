import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Optional, Callable

from src.normalization import GridEnvironmentNormalizer


def sinusoidal_encoding(positions: jnp.ndarray, d_model: int, max_freq: float = 0.1) -> jnp.ndarray:
    """
    Generate sinusoidal positional encodings.

    Args:
        positions: (N, d_pos) positions to encode
        d_model: dimension of the encoding
        max_freq: maximum frequency scale

    Returns:
        encodings: (N, d_model) sinusoidal encodings
    """
    N, d_pos = positions.shape
    d_per_pos = d_model // (2 * d_pos)  # dimensions per position coordinate

    # Create frequency bands
    freqs = jnp.arange(d_per_pos) / d_per_pos * max_freq

    # Compute encodings for each position dimension
    encodings = []
    for i in range(d_pos):
        pos_i = positions[:, i:i+1]  # (N, 1)
        angles = pos_i * freqs[None, :]  # (N, d_per_pos)
        encodings.append(jnp.sin(angles))
        encodings.append(jnp.cos(angles))

    # Concatenate all encodings
    return jnp.concatenate(encodings, axis=1)  # (N, ~d_model)


class SetONet(eqx.Module):
    """
    SetONet: Combines DeepOSet (branch) with DeepONet (trunk) architecture.
    The branch processes sets of (location, value) pairs using phi/rho networks
    with attention-based aggregation.
    """
    # Branch components (DeepOSet)
    phi: eqx.nn.MLP  # Encodes individual sensors
    aggregator: eqx.Module  # Mean pooling or AttentionPool
    rho: eqx.nn.MLP  # Post-aggregation network

    # Trunk component
    trunk: eqx.nn.MLP

    # Output layer
    use_bias: bool = eqx.field(static=True)
    bias: Optional[jnp.ndarray]

    # Dimensions
    p: int = eqx.field(static=True)  # Latent dimension
    output_size: int = eqx.field(static=True)
    aggregation_type: str = eqx.field(static=True)
    use_positional_encoding: bool = eqx.field(static=True)
    pos_encoding_dim: int = eqx.field(static=True)
    pos_encoding_max_freq: float = eqx.field(static=True)

    def __init__(
        self,
        input_size_src: int,      # Dimensionality of sensor location x_i
        output_size_src: int,     # Dimensionality of sensor value u(x_i)
        input_size_tgt: int,      # Dimensionality of trunk input y
        output_size_tgt: int,     # Dimensionality of final output G(u)(y)
        p: int = 32,              # Latent dimension or branch/trunk cross product
        phi_hidden_size: int = 128,
        phi_output_size: int = 128,
        rho_hidden_size: int = 128,
        trunk_hidden_size: int = 128,
        n_trunk_layers: int = 4,
        n_rho_layers: int = 4, 
        n_phi_layers: int = 4,
        activation: Callable = jax.nn.relu,
        use_bias: bool = True,
        aggregation_type: str = "attention",  # 'mean' or 'attention'
        attention_n_heads: int = 4,
        attention_n_tokens: int = 1,
        use_positional_encoding: bool = False,
        pos_encoding_dim: int = 64,
        pos_encoding_max_freq: float = 0.1,
        *,
        key: jr.PRNGKey,
    ):
        keys = jr.split(key, 5)

        self.p = p
        self.aggregation_type = aggregation_type
        self.use_bias = use_bias
        self.use_positional_encoding = use_positional_encoding
        self.output_size = output_size_tgt
        self.pos_encoding_dim = pos_encoding_dim
        self.pos_encoding_max_freq = pos_encoding_max_freq

        # Determine phi input dimension based on encoding
        if use_positional_encoding:
            # Validate encoding dimension
            if pos_encoding_dim % (2 * input_size_src) != 0:
                raise ValueError(f"For sinusoidal encoding, pos_encoding_dim ({pos_encoding_dim}) "
                               f"must be divisible by 2 * input_size_src ({2 * input_size_src}).")
            phi_input_dim = pos_encoding_dim + output_size_src
        else:
            # Combined input for phi: location + value
            phi_input_dim = input_size_src + output_size_src

        # Phi network: encodes individual sensors
        self.phi = eqx.nn.MLP(
            in_size=phi_input_dim,
            out_size=phi_output_size,
            width_size=phi_hidden_size,
            depth=n_phi_layers,
            activation=activation,
            key=keys[0],
        )

        # Aggregation layer
        if aggregation_type == "attention":
            self.aggregator = AttentionPool(
                d_model=phi_output_size,
                n_heads=attention_n_heads,
                n_tokens=attention_n_tokens,
                key=keys[1],
            )
            rho_input_dim = phi_output_size * attention_n_tokens
        else:  # mean pooling
            self.aggregator = MeanPool()
            rho_input_dim = phi_output_size

        # Rho network: processes aggregated representation
        self.rho = eqx.nn.MLP(
            in_size=rho_input_dim,
            #out_size=p * output_size_tgt,  # p basis functions * output dim
            out_size=p,  # p basis functions * output dim
            width_size=rho_hidden_size,
            depth=n_rho_layers,
            activation=activation,
            key=keys[2],
        )

        # Trunk network
        self.trunk = eqx.nn.MLP(
            in_size=input_size_tgt,
            out_size=p * output_size_tgt,  # p basis functions * output dim
            width_size=trunk_hidden_size,
            depth=n_trunk_layers,
            activation=activation,
            key=keys[3],
        )

        # Bias term
        if use_bias:
            self.bias = jr.normal(keys[4], (output_size_tgt,)) * 0.01
        else:
            self.bias = None

    def __call__(
        self,
        sensor_locations: jnp.ndarray,
        sensor_values: jnp.ndarray,
        target_location: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Args:
            sensor_locations: (N, input_size_src) locations of sensors
            sensor_values: (N, output_size_src) values at sensor locations
            target_location: (input_size_tgt,) query location
            mask: Optional (N,) boolean mask for valid sensors

        Returns:
            output: (output_size_tgt,) predicted value at target location
        """
        #print("SetOnet Atten __call__")
        #print(sensor_locations.shape)
        #print(sensor_values.shape)
        #print(target_location.shape)
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            # Apply sinusoidal encoding to sensor locations
            encoded_locations = sinusoidal_encoding(
                sensor_locations,
                self.pos_encoding_dim,
                self.pos_encoding_max_freq
            )
            # Combine encoded locations and values
            sensor_features = jnp.concatenate([encoded_locations, sensor_values], axis=-1)
        else:
            # Combine raw sensor locations and values
            sensor_features = jnp.concatenate([sensor_locations, sensor_values], axis=-1)

        # Encode each sensor
        #print("Encoding")
        #print(sensor_features.shape)
        encoded = jax.vmap(self.phi)(sensor_features)  # (N, phi_output_size)
        #print(encoded.shape)

        # Aggregate encoded sensors
        if self.aggregation_type == "attention":
            aggregated = self.aggregator(encoded, mask=mask)
        else:
            if mask is not None:
                # Apply mask for mean pooling
                encoded = jnp.where(mask[:, None], encoded, 0.0)
                aggregated = self.aggregator(encoded)
                # Normalize by number of valid sensors
                n_valid = mask.sum()
                aggregated = aggregated * encoded.shape[0] / jnp.maximum(n_valid, 1)
            else:
                aggregated = self.aggregator(encoded)

        # Process aggregated representation through rho
        branch_out = self.rho(aggregated)  # (p * output_size_tgt,)
        branch_out = jax.numpy.expand_dims(branch_out, 1)
        # Process target location through trunk
        trunk_out = self.trunk(target_location)  # (p * output_size_tgt,)

        trunk_out = trunk_out.reshape((self.p, self.output_size))

        # Element-wise multiplication and reshape
        #product = branch_out * trunk_out  # (p * output_size_tgt,)
        product = jnp.sum(branch_out * trunk_out,axis=0)

        # Reshape and sum over basis functions
        #product = product.reshape(self.p, -1)  # (p, output_size_tgt)
        #output = jnp.sum(product, axis=0)  # (output_size_tgt,)

        # Add bias if used
        if self.use_bias:
            output = product + self.bias

        return output


class MeanPool(eqx.Module):
    """Simple mean pooling aggregator."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: (N, d) features to pool

        Returns:
            pooled: (d,) mean of features
        """
        return jnp.mean(x, axis=0)


class AttentionPool(eqx.Module):
    """
    k-token multi-head attention aggregator (Set-Transformer style).
    If `n_tokens = 1` this is identical to single-token pooling.
    """
    query_tokens: jnp.ndarray
    attn: eqx.nn.MultiheadAttention
    n_tokens: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_tokens: int = 4,
        *,
        key: jr.PRNGKey,
    ):
        key_q, key_attn = jr.split(key)

        self.n_tokens = n_tokens
        # Initialize learnable query tokens
        self.query_tokens = jr.normal(key_q, (n_tokens, d_model)) * 0.02

        # Initialize multi-head attention
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d_model,
            key=key_attn,
        )

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Args:
            x: (N, d_model) encoded sensors/states
            mask: Optional (N,) boolean mask for valid positions

        Returns:
            pooled: (n_tokens * d_model,) flattened aggregated representation
        """
        # Expand query tokens for this input
        q = self.query_tokens  # (n_tokens, d_model)

        # Apply attention: queries attend to all input positions
        # x is keys and values
        if mask is not None:
            # Convert boolean mask to attention mask
            # True positions are attended to, False are masked out
            attn_mask = jnp.where(
                mask[None, :],  # (1, N)
                0.0,
                -jnp.inf
            ).repeat(self.n_tokens, axis=0)  # (n_tokens, N)
            pooled = self.attn(q, x, x, mask=attn_mask)  # (n_tokens, d_model)
        else:
            pooled = self.attn(q, x, x)  # (n_tokens, d_model)

        # Flatten the pooled tokens
        return pooled.flatten()  # (n_tokens * d_model,)


class SetONet1d(eqx.Module):
    """
    Simplified SetONet for 1D problems (like SimpleMaze value functions).
    Assumes scalar values at each state.
    """
    model: SetONet

    def __init__(
        self,
        num_states: int,
        p: int = 32,
        hidden_size: int = 128,
        depth: int = 2,
        activation: Callable = jax.nn.relu,
        aggregation_type: str = "attention",
        attention_n_tokens: int = 1,
        *,
        key: jr.PRNGKey,
    ):
        self.model = SetONet(
            input_size_src=1,  # 1D state index
            output_size_src=1,  # Scalar value
            input_size_tgt=1,  # 1D query state
            output_size_tgt=1,  # Scalar output
            p=p,
            phi_hidden_size=hidden_size,
            phi_output_size=hidden_size // 2,
            rho_hidden_size=hidden_size,
            trunk_hidden_size=hidden_size,
            n_trunk_layers=depth,
            activation=activation,
            aggregation_type=aggregation_type,
            attention_n_heads=4,
            attention_n_tokens=attention_n_tokens,
            use_bias=True,
            key=key,
        )

    def __call__(
        self,
        value_function: jnp.ndarray,  # (N,) value function over states
        query_state: jnp.ndarray,      # scalar or (1,) query state index
    ) -> jnp.ndarray:
        """
        Args:
            value_function: (N,) value function over all states
            query_state: scalar or (1,) state index to query

        Returns:
            value: scalar value at query state
        """
        N = value_function.shape[0]

        # Create state indices
        state_indices = jnp.arange(N).reshape(-1, 1).astype(jnp.float32) / N

        # Reshape value function
        values = value_function.reshape(-1, 1)

        # Ensure query_state is properly shaped
        if query_state.ndim == 0:
            query_state = jnp.array([query_state])
        query_state = query_state.astype(jnp.float32) / N

        # Call the model
        output = self.model(state_indices, values, query_state)

        return output.squeeze()


class NormalizedSetONet(eqx.Module):
    """
    SetONet wrapper that applies normalization to inputs.

    This wrapper normalizes states and actions before passing them to the SetONet model,
    and denormalizes the output actions. This is useful for training on physical systems
    where inputs have known bounds.

    Attributes:
        model: The underlying SetONet model
        normalizer: GridEnvironmentNormalizer for scaling inputs/outputs
        normalize_inputs: Whether to normalize inputs
        normalize_outputs: Whether to denormalize outputs
    """
    model: SetONet
    normalizer: GridEnvironmentNormalizer
    normalize_inputs: bool = eqx.field(static=True)
    normalize_outputs: bool = eqx.field(static=True)

    def __init__(
        self,
        model: SetONet,
        normalizer: GridEnvironmentNormalizer,
        normalize_inputs: bool = True,
        normalize_outputs: bool = True
    ):
        """
        Initialize normalized SetONet.

        Args:
            model: SetONet model to wrap
            normalizer: Normalizer for states and actions
            normalize_inputs: If True, normalize inputs before passing to model
            normalize_outputs: If True, denormalize outputs from model
        """
        self.model = model
        self.normalizer = normalizer
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs

    def __call__(
        self,
        sensor_locations: jnp.ndarray,
        sensor_values: jnp.ndarray,
        target_location: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Forward pass with normalization.

        For the typical use case in dynamics modeling:
        - sensor_locations: (state, action) pairs
        - sensor_values: next_states
        - target_location: (query_state, time) or (query_state, goal, time)
        - output: predicted action

        Args:
            sensor_locations: (N, state_dim + action_dim) sensor locations
            sensor_values: (N, state_dim) sensor values (next states)
            target_location: (state_dim + ...,) target query location
            mask: Optional (N,) boolean mask

        Returns:
            Predicted action (denormalized if normalize_outputs=True)
        """
        if self.normalize_inputs:
            # Split sensor_locations into states and actions
            # Assumes sensor_locations = [state, action]
            state_dim = sensor_values.shape[-1]

            # Normalize states in sensor locations
            sensor_states = sensor_locations[..., :state_dim]
            sensor_actions = sensor_locations[..., state_dim:]

            sensor_states_norm = self.normalizer.normalize_states(sensor_states)
            sensor_actions_norm = self.normalizer.normalize_actions(sensor_actions)
            sensor_locations_norm = jnp.concatenate([sensor_states_norm, sensor_actions_norm], axis=-1)

            # Normalize sensor values (next states)
            sensor_values_norm = self.normalizer.normalize_states(sensor_values)

            # Normalize target location (query state + possibly goal/time)
            # Only normalize the state portion
            target_state = target_location[..., :state_dim]
            target_rest = target_location[..., state_dim:]

            target_state_norm = self.normalizer.normalize_states(target_state)

            # If there's a goal state in target_location, normalize it too
            # Check if target_rest has goal dimensions (should be state_dim or state_dim+1 for time)
            if target_rest.shape[-1] > 1:
                # Has goal state + time
                target_goal = target_rest[..., :-1]
                target_time = target_rest[..., -1:]
                target_goal_norm = self.normalizer.normalize_states(target_goal)
                target_location_norm = jnp.concatenate([target_state_norm, target_goal_norm, target_time], axis=-1)
            elif target_rest.shape[-1] == 1:
                # Just time dimension
                target_location_norm = jnp.concatenate([target_state_norm, target_rest], axis=-1)
            else:
                # No additional dimensions
                target_location_norm = target_state_norm

            # Call model with normalized inputs
            output = self.model(sensor_locations_norm, sensor_values_norm, target_location_norm, mask)
        else:
            # No normalization
            output = self.model(sensor_locations, sensor_values, target_location, mask)

        if self.normalize_outputs:
            # Denormalize output (predicted action)
            output = self.normalizer.denormalize_actions(output)

        return output
