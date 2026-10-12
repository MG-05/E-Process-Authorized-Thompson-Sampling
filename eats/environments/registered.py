"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Mapping
import hashlib
import math
import numpy as np


GENERATOR_ID = "layer3_regret_atlas_generator_v7"


ATLAS_NAMESPACE = "layer3_atlas_v1"


K_VALUES = (4, 8, 16, 32, 64)


MECHANISMS = (
    "stationary",
    "isolated_incumbent_down",
    "isolated_challenger_up",
    "sparse_sustained_local_shock",
    "random_breakpoints",
    "global_switching",
    "rapid_dense_abrupt_switching",
    "mixed_abrupt_plus_drift",
    "asynchronous_per_arm_switching",
    "gradual_drift",
)


SOURCE_KINDS = (("random_environment", 96), ("sentinel", 32))


PROBABILITY_FLOOR = 0.02


PROBABILITY_CEILING = 0.98


MAX_GENERATION_ATTEMPTS = 32


DENSE_MINIMUM_PER_ARM_CHANGE = 1e-4


LATENT_APPLICABILITY: dict[str, tuple[str, ...]] = {
    "stationary": ("baseline_jitter_u", "gap_u", "phase_u"),
    "isolated_incumbent_down": (
        "baseline_jitter_u", "strength_u", "first_change_fraction"
    ),
    "isolated_challenger_up": (
        "baseline_jitter_u", "strength_u", "first_change_fraction"
    ),
    "sparse_sustained_local_shock": (
        "baseline_jitter_u", "strength_u", "gap_u", "affected_u",
        "first_change_fraction", "phase_u",
    ),
    "random_breakpoints": (
        "baseline_jitter_u", "strength_u", "gap_u", "recurrence_u",
        "first_change_fraction", "phase_u", "rank_u",
    ),
    "global_switching": (
        "baseline_jitter_u", "gap_u", "recurrence_u", "phase_u"
    ),
    "rapid_dense_abrupt_switching": (
        "baseline_jitter_u", "strength_u", "gap_u", "recurrence_u",
        "first_change_fraction", "phase_u",
    ),
    "mixed_abrupt_plus_drift": (
        "baseline_jitter_u", "strength_u", "gap_u", "drift_u",
        "first_change_fraction", "phase_u",
    ),
    "asynchronous_per_arm_switching": (
        "baseline_jitter_u", "strength_u", "gap_u", "phase_u"
    ),
    "gradual_drift": (
        "baseline_jitter_u", "gap_u", "drift_u", "cycle_u", "phase_u"
    ),
}


GENERATOR_SPEC: dict[str, Any] = {
    "generator_id": GENERATOR_ID,
    "random_draw_engine": "numpy.random.PCG64",
    "random_environments_are_iid": True,
    "finite_design_for_random_environments": False,
    "random_environment_count_per_stratum": 96,
    "sentinel_count_per_stratum": 32,
    "tapes_per_environment": 4,
    "bounded_attempt_count": MAX_GENERATION_ATTEMPTS,
    "contract_absolute_tolerance": 1e-12,
    "probability_support": [PROBABILITY_FLOOR, PROBABILITY_CEILING],
    "baseline_logit_centers": [-1.5, 0.0, 1.5],
    "baseline_logit_jitter": [-0.35, 0.35],
    "initial_oracle_gap": [0.04, 0.18],
    "isolated_target_old": [0.58, 0.84],
    "isolated_change_size": [0.16, 0.30],
    "isolated_comparison_offsets": [0.08, 0.20],
    "isolated_nuisance_tie_break_epsilon": 1e-8,
    "isolated_arm_roles": {
        "target_arm": 0,
        "comparison_arm": 1,
        "arm_indices_randomized": False,
    },
    "isolated_raw_support_lower_bound": 0.03999619,
    "isolated_raw_support_bound_derivation": (
        "1 - max(old_down) + min(change_size)/2 - max(comparison_offset) "
        "- (max(K)-3)*tie_break_epsilon - sentinel_count*uniqueness_epsilon"
    ),
    "first_change_fraction": [0.15, 0.75],
    "sparse_affected_fraction": [0.05, 0.20],
    "random_breakpoint_count": [2, 5],
    "global_segment_count": [2, 6],
    "rapid_segment_duration_expected_pulls": [4.0, 20.0],
    "rapid_minimum_per_arm_probability_change": DENSE_MINIMUM_PER_ARM_CHANGE,
    "mixed_drift_amplitude": [0.025, 0.10],
    "gradual_logit_amplitude": [0.15, 0.75],
    "gradual_cycles": [0.25, 2.0],
    "sentinel_coordinates": {
        "baseline_band": 3,
        "strength": 2,
        "first_change_fraction": [0.15, 0.35, 0.55, 0.75],
        "recurrence": 2,
    },
    "sentinel_corner_guarantee": (
        "marginal coverage of all baseline bands and both endpoints for every "
        "mechanism-applicable latent axis; not a full factorial design"
    ),
    "operative_latents_by_mechanism": LATENT_APPLICABILITY,
    "sentinel_uniqueness_epsilon": 1e-7,
    "deliberate_duplicate_sentinel_classes": [],
    "mechanism_laws": {
        "stationary": "one sampled arm vector repeated for all T rounds",
        "isolated_incumbent_down": "target a->b at tau; unchanged comparator c=(a+b)/2",
        "isolated_challenger_up": "target (1-a)->(1-b) at shared tau; comparator 1-c",
        "sparse_sustained_local_shock": "ceil(K*f), f in [0.05,0.20], clipped sustained shifts through T",
        "random_breakpoints": "2+min(3,floor(4u)) shifted breakpoints; one clipped arm shift by +/-(0.06+0.18s)",
        "global_switching": "2+min(4,floor(5u)) equal segments; arm vector cyclically permuted",
        "rapid_dense_abrupt_switching": (
            "first event tau, then round(K*(4+16u)); dense amplitude "
            "0.02+0.08s; every arm changes by at least 1e-4 at every event"
        ),
        "mixed_abrupt_plus_drift": "clipped sinusoidal probability drift plus one clipped -(0.06+0.20s) high-mean subset jump",
        "asynchronous_per_arm_switching": "one arm time in [0.15T,0.75T], alternating clipped +/-(0.05+0.20s)",
        "gradual_drift": "logit sinusoid with amplitude 0.15+0.60u and cycles 0.25+1.75u",
    },
    "matched_direction_construction": "Bernoulli complement",
    "dual_kl_solver_decimal_precision": 100,
    "dual_kl_residual_tolerance": "1e-12",
    "mean_table_encoding": "C-contiguous little-endian float64 with shape (T,K)",
    "retained_dense_tables": "one preregistered preflight fixture per stratum",
    "other_tables": "regenerated from frozen seed and generator identity; digest checked",
    "potential_outcomes_generated": False,
}


class EnvironmentBuildError(RuntimeError):
    """The outcome-free environment build violated a frozen contract."""


@dataclass(frozen=True)
class EnvironmentTask:
    mechanism: str
    n_arms: int
    budget: int
    source_kind: str
    source_index: int
    environment_seed: int
    fixture_path: str | None = None

    @property
    def horizon(self) -> int:
        return self.n_arms * self.budget

    @property
    def environment_id(self) -> str:
        return (
            f"{self.mechanism}:k{self.n_arms}:b{self.budget}:"
            f"{self.source_kind}:{self.source_index:03d}"
        )


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-array))


def _clip(table: np.ndarray) -> np.ndarray:
    return np.clip(table, PROBABILITY_FLOOR, PROBABILITY_CEILING).astype(
        np.float64, copy=False
    )


def _unit_latents(task: EnvironmentTask, attempt: int) -> dict[str, float | int]:
    nuisance_seed = task.environment_seed
    if task.mechanism in {
        "isolated_incumbent_down",
        "isolated_challenger_up",
    }:
        nuisance_seed = _seed(
            f"{ATLAS_NAMESPACE}|matched-nuisance|k{task.n_arms}|b{task.budget}|"
            f"{task.source_kind}|{task.source_index}"
        )
    if task.source_kind == "random_environment":
        sequence = np.random.SeedSequence([nuisance_seed, attempt])
        rng = np.random.default_rng(sequence)
        values = rng.random(10)
        baseline_band = int(rng.integers(0, 3))
    elif task.source_kind == "sentinel":
        index = task.source_index
        baseline_band = index % 3
        values = np.asarray(
            [
                ((index // 3) % 2),
                ((index // 6) % 4) / 3,
                ((index // 24) % 2),
                ((index * 7 + 1) % 31) / 30,
                ((index * 11 + 3) % 31) / 30,
                ((index * 13 + 5) % 31) / 30,
                ((index * 17 + 7) % 31) / 30,
                ((index * 19 + 9) % 31) / 30,
                ((index * 23 + 11) % 31) / 30,
                ((index * 5 + 15) % 31) / 30,
            ],
            dtype=float,
        )
    else:
        raise EnvironmentBuildError(f"unknown source kind: {task.source_kind}")
    first_fraction = (0.15, 0.35, 0.55, 0.75)[task.source_index % 4]
    if task.source_kind == "random_environment":
        first_fraction = 0.15 + 0.60 * float(values[1])
    return {
        "baseline_band": baseline_band,
        "nuisance_seed": nuisance_seed,
        "baseline_jitter_u": float(values[0]),
        "first_change_fraction": first_fraction,
        "strength_u": float(values[2]),
        "gap_u": float(values[3]),
        "affected_u": float(values[4]),
        "recurrence_u": float(values[5]),
        "drift_u": float(values[6]),
        "cycle_u": float(values[7]),
        "phase_u": float(values[8]),
        "rank_u": float(values[9]),
    }


def _base_means(task: EnvironmentTask, latent: Mapping[str, float | int]) -> np.ndarray:
    center = GENERATOR_SPEC["baseline_logit_centers"][int(latent["baseline_band"])]
    jitter = -0.35 + 0.70 * float(latent["baseline_jitter_u"])
    arms = np.arange(task.n_arms, dtype=float)
    phase = 2.0 * math.pi * float(latent["phase_u"])
    offsets = 0.55 * np.sin((arms + 1.0) * 1.618 + phase)
    offsets += np.linspace(0.25, -0.25, task.n_arms)
    values = _sigmoid(center + jitter + offsets)
    if task.source_kind == "sentinel":
        values[-1] -= (task.source_index + 1) * float(
            GENERATOR_SPEC["sentinel_uniqueness_epsilon"]
        )
    other_best_arm = 1 + int(np.argmax(values[1:]))
    values[1], values[other_best_arm] = values[other_best_arm], values[1]
    gap = 0.04 + 0.14 * float(latent["gap_u"])
    values[1:] = np.clip(
        values[1:],
        PROBABILITY_FLOOR + 0.01,
        PROBABILITY_CEILING - gap,
    )
    values[1:] -= np.arange(1, task.n_arms, dtype=float) * 1e-10
    best_other = float(np.max(values[1:]))
    values[0] = best_other + gap
    return _clip(values)


def _tau(task: EnvironmentTask, latent: Mapping[str, float | int]) -> int:
    return min(
        task.horizon - 2,
        max(1, int(round(task.horizon * float(latent["first_change_fraction"])))),
    )


def _constant_table(base: np.ndarray, horizon: int) -> np.ndarray:
    return np.broadcast_to(base, (horizon, len(base))).copy()


def _enforce_minimum_coordinate_change(
    previous: np.ndarray, proposed: np.ndarray, minimum: float
) -> np.ndarray:
    """Deterministically make every coordinate's absolute change at least minimum."""

    result = proposed.copy()
    deficient = np.flatnonzero(np.abs(result - previous) < minimum)
    for arm in deficient:
        intended_direction = 1.0 if result[arm] >= previous[arm] else -1.0
        candidate = previous[arm] + intended_direction * minimum
        if not PROBABILITY_FLOOR <= candidate <= PROBABILITY_CEILING:
            candidate = previous[arm] - intended_direction * minimum
        if not PROBABILITY_FLOOR <= candidate <= PROBABILITY_CEILING:
            raise EnvironmentBuildError("dense minimum change is infeasible")
        result[arm] = candidate
    return result


def _matched_table(
    task: EnvironmentTask, latent: Mapping[str, float | int]
) -> tuple[np.ndarray, dict[str, Any]]:
    old_min, old_max = GENERATOR_SPEC["isolated_target_old"]
    strength_min, strength_max = GENERATOR_SPEC["isolated_change_size"]
    near_offset, far_offset = GENERATOR_SPEC["isolated_comparison_offsets"]
    strength = strength_min + (strength_max - strength_min) * float(
        latent["strength_u"]
    )
    old_down = old_min + (old_max - old_min) * (
        0.15 * int(latent["baseline_band"]) / 2
        + 0.85 * float(latent["baseline_jitter_u"])
    )
    old_down = min(old_max, max(old_min, old_down))
    new_down = max(0.12, old_down - strength)
    comparison_down = 0.5 * (old_down + new_down)
    if not new_down < comparison_down < old_down:
        raise EnvironmentBuildError("matched role construction is infeasible")
    down = task.mechanism == "isolated_incumbent_down"
    old = old_down if down else 1.0 - old_down
    new = new_down if down else 1.0 - new_down
    comparison = comparison_down if down else 1.0 - comparison_down
    base = np.linspace(comparison - near_offset, comparison - far_offset, task.n_arms)
    base[0] = old
    base[1] = comparison
    tie_break = float(GENERATOR_SPEC["isolated_nuisance_tie_break_epsilon"])
    base[2:] -= np.arange(task.n_arms - 2) * tie_break
    if task.source_kind == "sentinel":
        base[-1] -= (task.source_index + 1) * float(
            GENERATOR_SPEC["sentinel_uniqueness_epsilon"]
        )
    raw_minimum = float(min(np.min(base), new))
    raw_maximum = float(max(np.max(base), new))
    if not (
        raw_minimum >= PROBABILITY_FLOOR
        and raw_maximum <= PROBABILITY_CEILING
    ):
        raise EnvironmentBuildError(
            "isolated construction would bind probability clipping"
        )
    derived_lower_bound = (
        1.0
        - old_max
        + strength_min / 2.0
        - far_offset
        - (max(K_VALUES) - 3) * tie_break
        - SOURCE_KINDS[1][1]
        * float(GENERATOR_SPEC["sentinel_uniqueness_epsilon"])
    )
    registered_lower_bound = float(
        GENERATOR_SPEC["isolated_raw_support_lower_bound"]
    )
    if abs(derived_lower_bound - registered_lower_bound) > 1e-15:
        raise EnvironmentBuildError("isolated registered support bound is stale")
    if raw_minimum < registered_lower_bound - 1e-12:
        raise EnvironmentBuildError("isolated derived support bound drift")
    base = _clip(base)
    tau = _tau(task, latent)
    table = _constant_table(base, task.horizon)
    table[tau:, 0] = new
    role = "incumbent" if down else "challenger"
    other_max = float(np.max(base[1:]))
    return table, {
        "event_times": [tau],
        "event_affected_arms": {str(tau): [0]},
        "affected_arms": [0],
        "target_arm": 0,
        "comparison_arm": 1,
        "arm_indices_randomized": False,
        "target_role": role,
        "old": float(old),
        "new": float(new),
        "pre_other_max": other_max,
        "post_other_max": other_max,
        "matching_construction": "complement",
        "probability_clipping_applied": False,
        "raw_probability_minimum": raw_minimum,
        "raw_probability_maximum": raw_maximum,
    }


def _generate_once(
    task: EnvironmentTask, latent: Mapping[str, float | int]
) -> tuple[np.ndarray, dict[str, Any]]:
    base = _base_means(task, latent)
    horizon, n_arms = task.horizon, task.n_arms
    tau = _tau(task, latent)
    mechanism = task.mechanism
    if mechanism == "stationary":
        return _constant_table(base, horizon), {
            "event_times": [], "event_affected_arms": {},
            "affected_arms": [], "target_arm": 0
        }
    if mechanism in {"isolated_incumbent_down", "isolated_challenger_up"}:
        return _matched_table(task, latent)
    if mechanism == "sparse_sustained_local_shock":
        table = _constant_table(base, horizon)
        affected_count = max(
            1,
            min(
                n_arms - 1,
                int(math.ceil((0.05 + 0.15 * float(latent["affected_u"])) * n_arms)),
            ),
        )
        affected = [0, *range(2, 1 + affected_count)]
        challenger = float(base[1])
        shock = 0.02 + 0.10 * float(latent["strength_u"])
        table[tau:, 0] = max(PROBABILITY_FLOOR, challenger - shock)
        for arm in affected[1:]:
            table[tau:, arm] = _clip(
                np.asarray([base[arm] - 0.05 * (arm + 1)])
            )[0]
        return table, {
            "event_times": [tau],
            "event_affected_arms": {str(tau): affected},
            "affected_arms": affected,
            "target_arm": 0,
            "target_role": "incumbent",
        }
    if mechanism == "random_breakpoints":
        count = 2 + min(3, int(math.floor(4 * float(latent["recurrence_u"]))))
        fractions = np.linspace(0.18, 0.82, count)
        shift = (float(latent["first_change_fraction"]) - 0.45) * 0.12
        events = sorted({
            min(
                int(math.floor(0.75 * horizon)),
                max(int(math.ceil(0.15 * horizon)), int(round(horizon * (f + shift)))),
            )
            for f in fractions
        })
        table = _constant_table(base, horizon)
        current = base.copy()
        affected_arms = []
        event_affected_arms: dict[str, list[int]] = {}
        for index, event in enumerate(events):
            arm = (index + int(latent["rank_u"] * n_arms)) % n_arms
            affected_arms.append(arm)
            event_affected_arms[str(event)] = [arm]
            current = current.copy()
            magnitude = 0.06 + 0.18 * float(latent["strength_u"])
            delta = magnitude if index % 2 == 0 else -magnitude
            proposed = _clip(np.asarray([current[arm] + delta]))[0]
            if proposed == current[arm]:
                proposed = _clip(np.asarray([current[arm] - delta]))[0]
            current[arm] = proposed
            table[event:] = current
        return table, {
            "event_times": events,
            "event_affected_arms": event_affected_arms,
            "affected_arms": sorted(set(affected_arms)),
            "target_arm": affected_arms[0],
        }
    if mechanism == "global_switching":
        segment_count = 2 + min(4, int(math.floor(5 * float(latent["recurrence_u"]))))
        events = [int(round(horizon * j / segment_count)) for j in range(1, segment_count)]
        table = _constant_table(base, horizon)
        for segment, start in enumerate([0, *events]):
            stop = events[segment] if segment < len(events) else horizon
            table[start:stop] = np.roll(base, segment % n_arms)
        return table, {
            "event_times": events,
            "event_affected_arms": {
                str(event): list(range(n_arms)) for event in events
            },
            "affected_arms": list(range(n_arms)),
            "target_arm": 0,
        }
    if mechanism == "rapid_dense_abrupt_switching":
        duration_pulls = 4.0 + 16.0 * float(latent["recurrence_u"])
        duration = max(2, int(round(n_arms * duration_pulls)))
        events = list(range(tau, horizon, duration))
        table = np.empty((horizon, n_arms), dtype=np.float64)
        starts = [0, *events]
        for segment, start in enumerate(starts):
            stop = events[segment] if segment < len(events) else horizon
            rolled = np.roll(base, segment % n_arms)
            dense_amplitude = 0.02 + 0.08 * float(latent["strength_u"])
            dense_shift = (
                np.zeros(n_arms)
                if segment == 0
                else dense_amplitude * np.sin(np.arange(n_arms) + segment)
            )
            segment_means = _clip(rolled + dense_shift)
            if segment:
                segment_means = _enforce_minimum_coordinate_change(
                    table[start - 1],
                    segment_means,
                    DENSE_MINIMUM_PER_ARM_CHANGE,
                )
            table[start:stop] = segment_means
        return table, {
            "event_times": events,
            "event_affected_arms": {
                str(event): list(range(n_arms)) for event in events
            },
            "affected_arms": list(range(n_arms)),
            "target_arm": 0,
            "segment_duration_expected_pulls": duration / n_arms,
        }
    if mechanism == "mixed_abrupt_plus_drift":
        time = np.arange(horizon, dtype=float) / max(1, horizon - 1)
        amplitude = 0.025 + 0.075 * float(latent["drift_u"])
        phase = 2 * math.pi * float(latent["phase_u"])
        arm_phase = phase + np.arange(n_arms)
        drift = amplitude * (
            np.sin(2 * math.pi * time[:, None] + arm_phase)
            - np.sin(arm_phase)[None, :]
        )
        table = _clip(base[None, :] + drift)
        affected_count = max(1, n_arms // 3)
        pre_event_order = np.argsort(-table[tau - 1], kind="stable")
        affected = [0, *[int(arm) for arm in pre_event_order if arm != 0]][
            :affected_count
        ]
        jump = 0.06 + 0.20 * float(latent["strength_u"])
        table[tau:, affected] = _clip(table[tau:, affected] - jump)
        return table, {
            "event_times": [tau],
            "event_affected_arms": {str(tau): affected},
            "affected_arms": affected,
            "target_arm": 0,
            "drift_amplitude": amplitude,
        }
    if mechanism == "asynchronous_per_arm_switching":
        table = _constant_table(base, horizon)
        arm_events: dict[str, list[int]] = {}
        event_affected_arms: dict[str, list[int]] = {}
        all_events = []
        for arm in range(n_arms):
            fraction = 0.15 + 0.60 * ((arm + 0.5 + float(latent["phase_u"])) / (n_arms + 1))
            event = min(horizon - 2, max(1, int(round(horizon * fraction))))
            magnitude = 0.05 + 0.20 * float(latent["strength_u"])
            delta = magnitude if arm % 2 else -magnitude
            table[event:, arm] = _clip(np.asarray([base[arm] + delta]))[0]
            arm_events[str(arm)] = [event]
            event_affected_arms.setdefault(str(event), []).append(arm)
            all_events.append(event)
        return table, {
            "event_times": sorted(set(all_events)),
            "event_affected_arms": event_affected_arms,
            "affected_arms": list(range(n_arms)),
            "arm_event_times": arm_events,
            "target_arm": 0,
        }
    if mechanism == "gradual_drift":
        time = np.arange(horizon, dtype=float) / max(1, horizon - 1)
        amplitude = 0.15 + 0.60 * float(latent["drift_u"])
        cycles = 0.25 + 1.75 * float(latent["cycle_u"])
        phase = 2 * math.pi * float(latent["phase_u"])
        base_logits = np.log(base / (1.0 - base))
        arm_phase = phase + np.arange(n_arms)
        path = base_logits[None, :] + amplitude * (
            np.sin(2 * math.pi * cycles * time[:, None] + arm_phase)
            - np.sin(arm_phase)[None, :]
        )
        return _clip(_sigmoid(path)), {
            "event_times": [],
            "event_affected_arms": {},
            "affected_arms": list(range(n_arms)),
            "target_arm": 0,
            "gradual_logit_amplitude": amplitude,
            "gradual_cycles": cycles,
        }
    raise EnvironmentBuildError(f"unknown mechanism: {mechanism}")


def validate_generated_environment(
    task: EnvironmentTask, table: np.ndarray, metadata: Mapping[str, Any]
) -> None:
    if table.shape != (task.horizon, task.n_arms):
        raise EnvironmentBuildError("mean-table geometry drift")
    if not np.all(np.isfinite(table)) or np.any(
        (table <= 0.0) | (table >= 1.0)
    ):
        raise EnvironmentBuildError("mean table is outside binary-reward eligibility")
    if np.any(table < PROBABILITY_FLOOR) or np.any(table > PROBABILITY_CEILING):
        raise EnvironmentBuildError("mean table escaped the frozen probability support")
    events = list(metadata.get("event_times", []))
    if events != sorted(set(int(value) for value in events)) or any(
        not 0 < int(value) < task.horizon for value in events
    ):
        raise EnvironmentBuildError("event metadata is invalid")
    raw_event_arms = metadata.get("event_affected_arms")
    if not isinstance(raw_event_arms, Mapping):
        raise EnvironmentBuildError("per-event affected-arm metadata is missing")
    try:
        event_arms = {
            int(event): sorted(set(int(arm) for arm in arms))
            for event, arms in raw_event_arms.items()
        }
    except (TypeError, ValueError) as error:
        raise EnvironmentBuildError("per-event affected-arm metadata is invalid") from error
    if sorted(event_arms) != events or any(
        not arms or any(not 0 <= arm < task.n_arms for arm in arms)
        for arms in event_arms.values()
    ):
        raise EnvironmentBuildError("per-event affected-arm census is invalid")
    aggregate_affected = sorted({arm for arms in event_arms.values() for arm in arms})
    if events and aggregate_affected != sorted(
        set(int(arm) for arm in metadata["affected_arms"])
    ):
        raise EnvironmentBuildError("aggregate and per-event affected arms disagree")
    if task.mechanism in {"stationary", "gradual_drift"} and events:
        raise EnvironmentBuildError("stationary/gradual environment has discrete events")
    if task.mechanism == "stationary" and not np.array_equal(
        table, np.broadcast_to(table[0], table.shape)
    ):
        raise EnvironmentBuildError("stationary environment is not stationary")
    initial = np.sort(table[0])
    realized_gap = float(initial[-1] - initial[-2])
    if not 0.04 - 1e-12 <= realized_gap <= 0.18 + 1e-12:
        raise EnvironmentBuildError("realized initial oracle gap is outside [0.04,0.18]")
    if task.mechanism not in {"stationary", "gradual_drift"} and not events:
        raise EnvironmentBuildError("abrupt mechanism has no materialized event")
    if events and not 0.15 <= events[0] / task.horizon <= 0.75:
        raise EnvironmentBuildError("first change is outside [0.15T,0.75T]")
    if any(np.array_equal(table[event - 1], table[event]) for event in events):
        raise EnvironmentBuildError("registered abrupt event does not change the table")
    pure_piecewise = {
        "isolated_incumbent_down",
        "isolated_challenger_up",
        "sparse_sustained_local_shock",
        "random_breakpoints",
        "global_switching",
        "rapid_dense_abrupt_switching",
        "asynchronous_per_arm_switching",
    }
    if task.mechanism in pure_piecewise:
        for event in events:
            changed = np.flatnonzero(table[event] != table[event - 1]).tolist()
            if changed != event_arms[event]:
                raise EnvironmentBuildError(
                    "registered per-event arms do not equal realized changed arms"
                )
    elif task.mechanism == "mixed_abrupt_plus_drift":
        for event in events:
            if any(table[event, arm] == table[event - 1, arm] for arm in event_arms[event]):
                raise EnvironmentBuildError("mixed abrupt component is not materialized")
    if task.mechanism in {"global_switching", "rapid_dense_abrupt_switching"}:
        if list(metadata.get("affected_arms", [])) != list(range(task.n_arms)):
            raise EnvironmentBuildError("dense/global affected-arm metadata is incomplete")
        changes = np.abs(table[np.asarray(events)] - table[np.asarray(events) - 1])
        if np.any(changes == 0.0):
            raise EnvironmentBuildError("dense/global event leaves an arm unchanged")
        if (
            task.mechanism == "rapid_dense_abrupt_switching"
            and np.any(changes < DENSE_MINIMUM_PER_ARM_CHANGE - 1e-12)
        ):
            raise EnvironmentBuildError("rapid event violates minimum per-arm change")
    if task.mechanism.startswith("isolated_"):
        tau = events[0]
        old = float(metadata["old"])
        new = float(metadata["new"])
        other = float(metadata["pre_other_max"])
        if task.mechanism.endswith("incumbent_down"):
            valid = old > other > new
        else:
            valid = old < other < new
        if (
            not valid
            or table[tau - 1, 0] != old
            or table[tau, 0] != new
            or np.any(table[:tau, 0] != old)
            or np.any(table[tau:, 0] != new)
        ):
            raise EnvironmentBuildError("isolated target role contract failed")
    if task.mechanism == "sparse_sustained_local_shock":
        affected = list(metadata["affected_arms"])
        if len(affected) > max(1, math.ceil(0.20 * task.n_arms)):
            raise EnvironmentBuildError("sparse mechanism is not sparse")


def bounded_generate(
    task: EnvironmentTask,
    *,
    accept: Callable[[np.ndarray, Mapping[str, Any]], bool] | None = None,
    max_attempts: int = MAX_GENERATION_ATTEMPTS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate under the frozen bounded filter; exhaustion is a hard failure."""

    reasons = []
    attempt_log = []
    for attempt in range(max_attempts):
        latent = _unit_latents(task, attempt)
        try:
            table, mechanism_metadata = _generate_once(task, latent)
            validate_generated_environment(task, table, mechanism_metadata)
            if accept is not None and not accept(table, mechanism_metadata):
                raise EnvironmentBuildError("external validity predicate rejected")
        except EnvironmentBuildError as error:
            reasons.append(str(error))
            attempt_log.append(
                {
                    "attempt": attempt + 1,
                    "latent": latent,
                    "accepted": False,
                    "rejection_reason": str(error),
                }
            )
            continue
        attempt_log.append(
            {
                "attempt": attempt + 1,
                "latent": latent,
                "accepted": True,
                "rejection_reason": None,
            }
        )
        metadata = {
            **mechanism_metadata,
            "latent": latent,
            "attempt_count": attempt + 1,
            "rejection_reasons": reasons,
            "attempt_log": attempt_log,
            "accepted_after_bounded_filter": True,
        }
        return table, metadata
    raise EnvironmentBuildError(
        f"bounded generation exhausted for {task.environment_id}: {reasons[-3:]}"
    )
