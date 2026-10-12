"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from typing import Any
from typing import Sequence
import hashlib


CALIBRATION_NAMESPACE_ID = "e_value_restart_fresh_iid_evaluation_v8"


CONFIRMATION_NAMESPACE_ID = "e_value_restart_branch_b_confirmation_v4"


MATCHED_KL_NAMESPACE_ID = "e_value_restart_branch_b_matched_kl_v4"


PREFLIGHT_NAMESPACE_ID = "e_value_workshop_phase0_preflight_v3"


UNIT_TEST_NAMESPACE_ID = "e_value_workshop_phase0_unit_test_v2"


SEED_SCHEME_ID = "sha256_namespace_entropy_spawn_key_pcg64_v2"


CALIBRATION_GEOMETRIES = ((4, 2_500), (32, 20_000), (8, 50_000))


def seed_descriptor(namespace: str, indices: Sequence[int], stream_id: int) -> dict[str, Any]:
    permitted = {
        CALIBRATION_NAMESPACE_ID, CONFIRMATION_NAMESPACE_ID, MATCHED_KL_NAMESPACE_ID,
        PREFLIGHT_NAMESPACE_ID, UNIT_TEST_NAMESPACE_ID,
    }
    if namespace not in permitted:
        raise ValueError("unregistered Phase-0-v2 namespace.")
    if not indices or any(int(i) < 0 for i in indices):
        raise ValueError("indices must be nonempty and nonnegative.")
    if stream_id not in {0, 1, 2, 3}:
        raise ValueError("unregistered stream ID.")
    return {
        "scheme": SEED_SCHEME_ID,
        "namespace": namespace,
        "entropy": int.from_bytes(hashlib.sha256(namespace.encode()).digest(), "big"),
        "spawn_key": [*(int(i) for i in indices), int(stream_id)],
        "pool_size": 4,
        "bit_generator": "PCG64",
    }
