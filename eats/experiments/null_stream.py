"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from eats.experiments.null_seeds import CALIBRATION_GEOMETRIES
from eats.experiments.null_seeds import seed_descriptor
import numpy as np


def _rng(namespace: str, geometry_index: int, trajectory_index: int, stream_id: int) -> np.random.Generator:
    descriptor = seed_descriptor(
        namespace, (geometry_index, trajectory_index), stream_id
    )
    sequence = np.random.SeedSequence(
        int(descriptor["entropy"]),
        spawn_key=tuple(int(value) for value in descriptor["spawn_key"]),
        pool_size=int(descriptor["pool_size"]),
    )
    return np.random.Generator(np.random.PCG64(sequence))


def _prior_predictive_table(
    geometry_index: int,
    trajectory_index: int,
    *,
    namespace: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_arms, horizon = CALIBRATION_GEOMETRIES[geometry_index]
    environment_rng = _rng(namespace, geometry_index, trajectory_index, 2)
    reward_rng = _rng(namespace, geometry_index, trajectory_index, 0)
    probabilities = environment_rng.beta(1.0, 1.0, size=n_arms)
    potential = (
        reward_rng.random((n_arms, horizon)) < probabilities[:, None]
    ).astype(np.int8)
    return probabilities, potential


def _optimistic_ts(
    potential: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_arms, horizon = potential.shape
    alpha = np.ones(n_arms, dtype=float)
    beta = np.ones(n_arms, dtype=float)
    actions = np.empty(horizon, dtype=np.int64)
    rewards = np.empty(horizon, dtype=np.int8)
    for timestep in range(horizon):
        posterior_mean = alpha / (alpha + beta)
        arm = int(np.argmax(np.maximum(rng.beta(alpha, beta), posterior_mean)))
        reward = int(potential[arm, timestep])
        actions[timestep] = arm
        rewards[timestep] = reward
        alpha[arm] += reward
        beta[arm] += 1 - reward
    return actions, rewards
