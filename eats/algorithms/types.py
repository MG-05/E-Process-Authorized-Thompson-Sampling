"""Paper-only extraction; original symbol hashes are in provenance/code.json."""
from __future__ import annotations

from numpy.typing import NDArray
from typing import TypeAlias
from typing import TypedDict
import numpy as np


FloatArray: TypeAlias = NDArray[np.float64]


IntArray: TypeAlias = NDArray[np.int64]


BoolArray: TypeAlias = NDArray[np.bool_]


class GlobalCTSRunDiagnostics(TypedDict, total=False):
    _record_played_arms: bool
    _record_final_state: bool
    implementation_id: str
    max_particle_support: int
    resampling_count: int
    resample_target_fraction: float
    peak_persistent_model_state_bytes: int
    reward_output_bytes: int
    final_weights: FloatArray
    final_alpha: FloatArray
    final_beta: FloatArray
    played_arms: IntArray


class MUCBRunDiagnostics(TypedDict, total=False):
    _record_played_arms: bool
    implementation_id: str
    played_arms: IntArray
    forced_cycle: int
    restart_count: int
    restart_rounds: IntArray


class AFFTSRunDiagnostics(TypedDict, total=False):
    _record_played_arms: bool
    _record_state_history: bool
    implementation_id: str
    played_arms: IntArray
    lambda_history: FloatArray
    final_lambda: FloatArray
    final_m: FloatArray
    final_w: FloatArray
    final_m_dot: FloatArray
    final_w_dot: FloatArray
    final_last_selected: IntArray


class GLRklUCBRunDiagnostics(TypedDict, total=False):
    _record_played_arms: bool
    _record_final_histories: bool
    _record_allocated_state: bool
    implementation_id: str
    reference_implementation_id: str
    optimization_reference_implementation_id: str
    index_computation: str
    resolved_delta: float
    restart_mode: str
    episode_count: int
    restart_count: int
    forced_exploration_count: int
    detector_call_count: int
    split_candidates_examined: int
    peak_selected_history_length: int
    peak_retained_observations: int
    final_retained_observations: int
    history_itemsize_bytes: int
    state_representation: str
    final_logical_model_state_bytes: int
    peak_logical_model_state_bytes: int
    final_allocated_model_state_bytes: int
    peak_allocated_model_state_bytes: int
    played_arms: IntArray
    forced_exploration: BoolArray
    restart_rounds: IntArray
    restart_arms: IntArray
    final_restart_times: IntArray
    final_history_lengths: IntArray
    final_histories: tuple[IntArray, ...]
    zero_mean_index_evaluations: int
    one_mean_index_evaluations: int


class CUSUMUCBRunDiagnostics(TypedDict, total=False):
    _record_played_arms: bool
    implementation_id: str
    played_arms: IntArray
    restart_count: int
    restart_rounds: IntArray
    restart_arms: IntArray
    forced_warmup_count: int
    final_valid_pulls: FloatArray
    final_detector_counts: IntArray
    final_countdowns: IntArray


class GammaSWGTSRunDiagnostics(TypedDict, total=False):
    _record_played_arms: bool
    implementation_id: str
    played_arms: IntArray
    forced_zero_count_plays: int
    final_window_pulls: IntArray
    final_window_successes: FloatArray
