"""Rebuild exactly the three figures included by paper/main.tex and appendix.tex.
Plotting helpers are extracted from the publication builder; data now come
from the compact paper-only release and independently recomputed summaries.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from collections.abc import Callable, Mapping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from eats.io import ROOT
from eats.environments.replay import build_scale_environment
OUT = ROOT / 'results/figures'
PRIMARY_ENVIRONMENT_FIXTURES = ROOT / 'data/fixtures'


OUTPUT_DPI = 1000

PNG_METADATA = {"Software": "E_Values_Workshop_2026/make_figures.py"}

INK = "#36434A"

SUMMARY_BLUE = "#35618D"

SUMMARY_CORAL = "#D9795F"

SUMMARY_TEAL = "#2B887F"

SUMMARY_PLUM = "#8066A5"

SUMMARY_SLATE = "#A8B1BA"

SUMMARY_ZERO = "#59656E"

SUMMARY_POSITIVE_FILL = "#F1F6FA"

SUMMARY_NEGATIVE_FILL = "#FBF2EF"

ARM_COLORS = (SUMMARY_BLUE, SUMMARY_CORAL, SUMMARY_TEAL, SUMMARY_PLUM)

PRIMARY_ATLAS_K = 4

PRIMARY_ATLAS_B = 625

PRIMARY_MECHANISM_NAMES = {
    "isolated_incumbent_down": "Isolated incumbent down",
    "isolated_challenger_up": "Isolated challenger up",
    "sparse_sustained_local_shock": "Sparse sustained shock",
    "random_breakpoints": "Random breakpoints",
    "global_switching": "Global switching",
    "rapid_dense_abrupt_switching": "Rapid dense switching",
    "mixed_abrupt_plus_drift": "Abrupt change + drift",
    "asynchronous_per_arm_switching": "Asynchronous per-arm",
    "gradual_drift": "Gradual drift",
}

HISTORICAL_ENVIRONMENT_SEED = 1_840_201

HISTORICAL_ATLAS_K = 4

HISTORICAL_ENVIRONMENT_SHA256 = {
    4: {
        "slow": "2650810f08b59a9ab90b550da90bd45a2f5d66078a8f440a93b2dd5cb4ec4617",
        "fast": "2cc436099fb620232131f273d1698b9bae745eb5cf0ad8e909eaa237371020f0",
        "abrupt": "0aa24092b9c63e3af92ab38292ee2fd0e72f395d985b159ae995e471a1f9db77",
        "mixed": "720933055f6debbc6063dc1445790fb21ebeaaaa62dc91e1c324c8b2af867383",
        "random_breakpoints": "4078d59bce1b6e9fb1fef7864d8abd454d8154a67b5ad3285809a4191260893f",
        "random_drift": "39a02245616da20a7074cdfac4ad0d95cc7990454655fd6b78960005c1ffa160",
        "global_switching": "bc9a8283802faf8ebe948aec03e478cf770e5ef3fa5186e3364332bdd3164700",
        "per_arm_switching": "d1f131619c11e5cfa4dfca7f6fd368034f46ffd402c8f27a1769de2c6d82200c",
        "local_shocks": "cdde65aa03452d4eeff322b9f57b15422d4875856daecb87c5695c242e6b8c21",
    },
    32: {
        "slow": "51c23314c28691b9e1f29d1bafed39a3a594ae3b3ba5275dfd7d4026b9a5feca",
        "fast": "041fe4f7c3f76d114721e300e30e64574b4027a81230a4a47028f5b54f67b850",
        "abrupt": "95fa01b754fc9e0e83a041efbe375f3dace740f3dec3229b508107f96554aad3",
        "mixed": "3dec7e18300000eb55d65f8fca829add8c4f10f33ac1e42769cf23160bc4ec4a",
        "random_breakpoints": "ab10e4088b9efa256949c33b0b49fa9b6ee4b908b5a4de620ea1fc5cfa6db8a5",
        "random_drift": "51d238ea5824da499e772dbd9e71e89a04b80de2d1ae1353fe83c4916f47e05c",
        "global_switching": "bd54943abd3890ff17c63d48c0a583f0313603ccaffb8ccf741100fa509f539c",
        "per_arm_switching": "1c0826687cc8f003563e7835e8bcd0c594f3961641175b741089942202b1a36d",
        "local_shocks": "c51fa5c7212fea5e73de134f0ef273155aac7c1013786038489d284d3accda00",
    },
}

HISTORICAL_DISPLAY_NAMES = {
    "slow": "Slow sine",
    "fast": "Fast sine",
    "abrupt": "Abrupt cycle",
    "global_switching": "Global switching",
    "per_arm_switching": "Per-arm switching",
    "local_shocks": "Local shocks",
    "random_breakpoints": "Random breakpoints",
    "random_drift": "Random drift",
    "mixed": "Mixed regime",
}

REPLAY_FAMILY_ORDER = (
    "abrupt",
    "fast",
    "global_switching",
    "local_shocks",
    "mixed",
    "per_arm_switching",
    "random_breakpoints",
    "random_drift",
    "slow",
)

def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _save_png(figure: plt.Figure, filename: str) -> None:
    """Write one deterministic, publication-resolution raster figure."""
    figure.savefig(
        OUT / filename,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata=PNG_METADATA,
    )
    plt.close(figure)

def _primary_environment_means(mechanism: str) -> np.ndarray:
    """Load and verify one frozen primary-design K=4 mean table."""

    path = PRIMARY_ENVIRONMENT_FIXTURES / (
        f"{mechanism}_k{PRIMARY_ATLAS_K}_b{PRIMARY_ATLAS_B}.npy"
    )
    identity = _load(path.with_suffix(".npy.sha256.json"))
    if _sha256_file(path) != identity["file_sha256"]:
        raise RuntimeError(f"Primary environment file identity drift: {mechanism}")
    means = np.load(path, allow_pickle=False)
    expected_shape = (PRIMARY_ATLAS_K * PRIMARY_ATLAS_B, PRIMARY_ATLAS_K)
    if means.shape != expected_shape or list(means.shape) != identity["shape"]:
        raise RuntimeError(f"Unexpected primary environment geometry: {mechanism}")
    digest = hashlib.sha256(
        np.ascontiguousarray(means, dtype="<f8").tobytes()
    ).hexdigest()
    if digest != identity["mean_table_sha256"]:
        raise RuntimeError(f"Primary mean-table identity drift: {mechanism}")
    return np.asarray(means, dtype=float)

def _historical_environment_means(family: str, n_arms: int) -> np.ndarray:
    """Return and verify one frozen historical-suite mean table."""

    environment = build_scale_environment(
        family=family,
        n_arms=n_arms,
        seed=HISTORICAL_ENVIRONMENT_SEED,
    )
    means = np.asarray(environment.means_tk, dtype=float)
    if means.shape != (625 * n_arms, n_arms):
        raise RuntimeError(f"Unexpected historical environment geometry: {family}")
    digest = hashlib.sha256(np.ascontiguousarray(means).tobytes()).hexdigest()
    if digest != HISTORICAL_ENVIRONMENT_SHA256[n_arms][family]:
        raise RuntimeError(f"Historical mean-table identity drift: {family}")
    return means

def _environment_atlas_figure(
    *,
    display_names: Mapping[str, str],
    load_means: Callable[[str], np.ndarray],
    horizon: int,
    filename: str,
    x_windows: Mapping[str, tuple[int, int]] | None = None,
) -> None:
    """Render nine four-arm environments with a shared visual grammar."""

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.15, 5.55),
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        top=0.96,
        bottom=0.16,
        hspace=0.62,
        wspace=0.44,
    )
    for axis, key in zip(axes.flat, display_names, strict=True):
        means = load_means(key)
        if means.shape != (horizon, len(ARM_COLORS)):
            raise RuntimeError(f"Unexpected four-arm atlas geometry: {key}")
        time = np.arange(means.shape[0])
        for arm in range(means.shape[1]):
            axis.plot(
                time,
                means[:, arm],
                color=ARM_COLORS[arm],
                alpha=0.94,
                lw=0.90,
                rasterized=True,
            )
        axis.plot(
            time,
            means.max(axis=1),
            color=INK,
            ls="--",
            lw=1.15,
            zorder=5,
            rasterized=True,
        )
        start, stop = (x_windows or {}).get(key, (0, horizon))
        midpoint = start + (stop - start) // 2
        axis.set_title(display_names[key], fontsize=8.0, pad=2.0)
        axis.set_xlim(start, stop)
        axis.set_ylim(0, 1)
        axis.set_xticks(
            [start, midpoint, stop],
            [_format_round_tick(value) for value in (start, midpoint, stop)],
        )
        axis.set_yticks([0, 0.5, 1.0])
        axis.set_xlabel("Round $t$", fontsize=7.6, labelpad=1.0)
        axis.set_ylabel("Bernoulli mean", fontsize=7.6, labelpad=1.5)
        axis.tick_params(labelsize=6.8, width=0.45, pad=1.5)
        axis.grid(True, color=SUMMARY_SLATE, alpha=0.24, lw=0.35)
        for spine in axis.spines.values():
            spine.set_color(SUMMARY_ZERO)
            spine.set_linewidth(0.5)
    handles = [
        *[
            plt.Line2D([], [], color=color, lw=1.25, label=f"Arm {arm}")
            for arm, color in enumerate(ARM_COLORS, start=1)
        ],
        plt.Line2D([], [], color=INK, ls="--", lw=1.2, label="Oracle mean"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        fontsize=7.2,
    )
    _save_png(fig, filename)

def _format_round_tick(value: int) -> str:
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return str(value)

def primary_environment_atlas_figure() -> None:
    """Show one frozen K=4 instance of every primary nonstationary mechanism."""

    _environment_atlas_figure(
        display_names=PRIMARY_MECHANISM_NAMES,
        load_means=_primary_environment_means,
        horizon=PRIMARY_ATLAS_K * PRIMARY_ATLAS_B,
        filename="primary_environment_atlas.png",
    )

def _historical_atlas_means(family: str) -> np.ndarray:
    return _historical_environment_means(family, HISTORICAL_ATLAS_K)

def historical_environment_atlas_figure() -> None:
    """Show one frozen K=4 instance of every retrospective family."""

    _environment_atlas_figure(
        display_names=HISTORICAL_DISPLAY_NAMES,
        load_means=_historical_atlas_means,
        horizon=625 * HISTORICAL_ATLAS_K,
        filename="historical_environment_atlas.png",
        x_windows={"fast": (0, 1_250)},
    )

def authorization_and_rank_summary_figure() -> None:
    from eats.analysis import compute, verify_paper
    report = compute()
    verify_paper(report)
    registered_by_k = report['registered_by_k']
    replay_by_k = report['replay_by_k']
    ranks = {(r['family'], r['K']): r['rank'] for r in report['replay_cells'] if r['policy'] == 'e-ats'}

    expected_ranks = {
        ("abrupt", 4): 14,
        ("abrupt", 32): 8,
        ("fast", 4): 13,
        ("fast", 32): 8,
        ("global_switching", 4): 14,
        ("global_switching", 32): 9,
        ("local_shocks", 4): 12,
        ("local_shocks", 32): 5,
        ("mixed", 4): 13,
        ("mixed", 32): 1,
        ("per_arm_switching", 4): 14,
        ("per_arm_switching", 32): 11,
        ("random_breakpoints", 4): 14,
        ("random_breakpoints", 32): 10,
        ("random_drift", 4): 6,
        ("random_drift", 32): 2,
        ("slow", 4): 13,
        ("slow", 32): 8,
    }
    if ranks != expected_ranks:
        raise RuntimeError("Recomputed 17-policy ranks disagree with the paper")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.35),
        gridspec_kw={"width_ratios": [1.04, 1.22]},
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.25, wspace=0.50)
    left, right = axes
    fig.suptitle(
        "Authorization effects and replay-suite ranks",
        fontsize=10.0,
        color=INK,
        y=0.965,
    )

    registered_ks = np.asarray(sorted(registered_by_k), dtype=float)
    replay_ks = np.asarray(sorted(replay_by_k), dtype=float)
    left.axhspan(0.0, 0.043, color=SUMMARY_POSITIVE_FILL, zorder=-2)
    left.axhspan(-0.020, 0.0, color=SUMMARY_NEGATIVE_FILL, zorder=-2)
    left.axhline(0.0, color=SUMMARY_ZERO, lw=0.85, ls="--", zorder=0)
    left.plot(
        registered_ks,
        [registered_by_k[int(k)] for k in registered_ks],
        color=SUMMARY_BLUE,
        marker="o",
        ms=4.0,
        lw=1.35,
        label="Registered suite",
    )
    left.plot(
        replay_ks,
        [replay_by_k[int(k)] for k in replay_ks],
        color=SUMMARY_CORAL,
        marker="s",
        ms=4.0,
        lw=1.35,
        label="Replay suite",
    )
    left.set_xscale("log", base=2)
    left.set_xticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
    left.set_xlim(3.5, 72)
    left.set_ylim(-0.020, 0.043)
    left.set_title("Authorization effect by arm count", fontsize=8.8, pad=4)
    left.set_xlabel(r"Number of arms $K$", fontsize=7.8)
    left.set_ylabel("Open-ramp regret minus e-ATS regret", fontsize=7.8)
    left.tick_params(labelsize=7.0, width=0.5)
    left.grid(True, alpha=0.18, lw=0.4)
    left.legend(frameon=False, fontsize=6.9, loc="upper left")

    y = np.arange(len(REPLAY_FAMILY_ORDER))
    for yi, family in zip(y, REPLAY_FAMILY_ORDER, strict=True):
        rank_4 = ranks[(family, 4)]
        rank_32 = ranks[(family, 32)]
        connector = SUMMARY_SLATE
        linewidth = 0.9
        if family == "mixed":
            connector = SUMMARY_PLUM
            linewidth = 1.5
        elif family == "random_drift":
            connector = SUMMARY_TEAL
            linewidth = 1.5
        right.plot([rank_4, rank_32], [yi, yi], color=connector, lw=linewidth, alpha=0.82)
        right.scatter(
            rank_4,
            yi,
            marker="o",
            s=27,
            facecolors="white",
            edgecolors=SUMMARY_BLUE,
            linewidths=1.0,
            zorder=3,
        )
        right.scatter(
            rank_32,
            yi,
            marker="s",
            s=27,
            color=SUMMARY_CORAL,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
    right.set_yticks(
        y,
        [HISTORICAL_DISPLAY_NAMES[family] for family in REPLAY_FAMILY_ORDER],
        fontsize=6.8,
    )
    right.invert_yaxis()
    right.set_xlim(0.25, 17.75)
    right.set_xticks([1, 4, 8, 12, 17])
    right.set_title("e-ATS rank within each replay family", fontsize=8.8, pad=4)
    right.set_xlabel("Rank among 17 methods (lower is better)", fontsize=7.8)
    right.set_ylabel("Replay-suite family", fontsize=7.8)
    right.tick_params(axis="x", labelsize=7.0, width=0.5)
    right.tick_params(axis="y", width=0.5, pad=2.0)
    right.grid(True, axis="x", alpha=0.18, lw=0.4)
    rank_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=SUMMARY_BLUE,
            linestyle="none",
            label=r"$K=4,\ T=2{,}500$",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            markersize=4.5,
            markerfacecolor=SUMMARY_CORAL,
            markeredgecolor="white",
            linestyle="none",
            label=r"$K=32,\ T=20{,}000$",
        ),
    ]
    right.legend(
        handles=rank_handles,
        frameon=False,
        fontsize=6.6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        columnspacing=1.4,
    )
    for spine in (*left.spines.values(), *right.spines.values()):
        spine.set_linewidth(0.5)

    _save_png(fig, "authorization_and_rank_summary.png")

def build(destination):
    global OUT
    OUT = Path(destination)
    OUT.mkdir(parents=True, exist_ok=True)
    primary_environment_atlas_figure()
    historical_environment_atlas_figure()
    authorization_and_rank_summary_figure()
    return OUT
