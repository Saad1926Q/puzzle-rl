#!/usr/bin/env python3
"""Generate publication-ready charts from archived Qwen 8-puzzle evaluations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "graphs"
COLORS = {
    "solved": "#2A9D8F",
    "illegal": "#E76F51",
    "malformed": "#F4A261",
    "truncated": "#6C77B8",
    "timeout": "#8D99AE",
    "pass": "#264653",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "font.size": 10,
            "legend.frameon": False,
            "grid.alpha": 0.22,
        }
    )


def easy_nonthinking_comparison() -> None:
    paths = [
        DATA_DIR / "qwen3.5-0.8b/qwen3.5-0.8b-nonthinking-easy-8rollouts.json",
        DATA_DIR / "qwen3.5-2b/qwen3.5-2b-nonthinking-easy-8rollouts.json",
        DATA_DIR / "qwen3.5-4b/qwen3.5-4b-nonthinking-easy-8rollouts.json",
    ]
    labels = ["0.8B", "2B", "4B"]
    results = [load_json(path)["summary"] for path in paths]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = np.arange(len(labels))
    solved = np.array([result["solved_rate"] for result in results])
    passed = np.array([result["pass@k"] for result in results])
    width = 0.34
    bars_a = left.bar(
        x - width / 2,
        solved,
        width,
        label="Episode solved rate",
        color=COLORS["solved"],
    )
    bars_b = left.bar(
        x + width / 2, passed, width, label="Puzzle pass@8", color=COLORS["pass"]
    )
    left.bar_label(bars_a, labels=[f"{value:.1%}" for value in solved], padding=3)
    left.bar_label(bars_b, labels=[f"{value:.1%}" for value in passed], padding=3)
    left.set_title("Capability on Easy Puzzles (Depth 1–5)")
    left.set_ylabel("Rate")
    left.set_xticks(x, labels)
    left.set_ylim(0, 0.78)
    left.yaxis.set_major_formatter(PercentFormatter(1))
    left.legend(loc="upper left")

    categories = ["solved", "illegal", "malformed", "truncated", "timeout"]
    bottoms = np.zeros(len(labels))
    for category in categories:
        values = np.array([result[f"{category}_rate"] for result in results])
        right.bar(
            x,
            values,
            bottom=bottoms,
            label=category.title(),
            color=COLORS[category],
            width=0.62,
        )
        bottoms += values
    right.set_title("Episode Outcomes")
    right.set_ylabel("Share of 120 episodes")
    right.set_xticks(x, labels)
    right.set_ylim(0, 1)
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

    fig.suptitle("Qwen3.5 Non-Thinking Model Comparison", fontsize=18, weight="bold")
    fig.text(
        0.5,
        -0.02,
        "15 puzzles × 8 rollouts; max_tokens=128; corrected goal prompt and vLLM thinking flag",
        ha="center",
        color="#555555",
    )
    fig.tight_layout()
    save(fig, "model_comparison_easy")


def qwen4_depth_profile() -> None:
    path = (
        DATA_DIR / "qwen3.5-4b/qwen3.5-4b-nonthinking-easy-8rollouts.trajectories.json"
    )
    episodes = load_json(path)["episodes"]
    by_depth: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"episodes": 0, "solved": 0, "puzzles": set(), "passed": set()}
    )
    for episode in episodes:
        depth = episode["optimal_length"]
        bucket = by_depth[depth]
        bucket["episodes"] += 1
        bucket["puzzles"].add(episode["id"])
        if episode["outcome"] == "solved":
            bucket["solved"] += 1
            bucket["passed"].add(episode["id"])

    depths = sorted(by_depth)
    solved_rates = [by_depth[d]["solved"] / by_depth[d]["episodes"] for d in depths]
    pass_rates = [
        len(by_depth[d]["passed"]) / len(by_depth[d]["puzzles"]) for d in depths
    ]
    counts = [by_depth[d]["episodes"] for d in depths]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(
        depths,
        solved_rates,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color=COLORS["solved"],
        label="Episode solved rate",
    )
    ax.plot(
        depths,
        pass_rates,
        marker="s",
        linewidth=2.5,
        markersize=7,
        color=COLORS["pass"],
        label="Puzzle pass@8",
    )
    for depth, solved_rate, count in zip(depths, solved_rates, counts, strict=True):
        ax.annotate(
            f"{solved_rate:.0%}\nn={count}",
            (depth, solved_rate),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#444444",
        )
    ax.set_title("Qwen3.5-4B Capability Frontier")
    ax.set_xlabel("Exact optimal solution depth")
    ax.set_ylabel("Rate")
    ax.set_xticks(depths)
    ax.set_ylim(-0.12, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "qwen4_depth_profile")


def qwen4_reward_distribution() -> None:
    files = {
        "Easy\n(depth 1–5)": DATA_DIR
        / "qwen3.5-4b/qwen3.5-4b-nonthinking-easy-8rollouts.trajectories.json",
        "Medium\n(depth 6–15)": DATA_DIR
        / "qwen3.5-4b/qwen3.5-4b-nonthinking-medium-8rollouts.trajectories.json",
    }
    labels = list(files)
    rewards = [
        [episode["reward"] for episode in load_json(path)["episodes"]]
        for path in files.values()
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    parts = ax.violinplot(rewards, showmeans=True, showmedians=True, widths=0.72)
    for body, color in zip(
        parts["bodies"], [COLORS["solved"], COLORS["illegal"]], strict=True
    ):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.72)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("#333333")
    rng = np.random.default_rng(42)
    for index, values in enumerate(rewards, start=1):
        jitter = rng.normal(index, 0.035, len(values))
        ax.scatter(jitter, values, s=10, alpha=0.25, color="#222222", linewidths=0)
    ax.axhline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_title("Qwen3.5-4B Episode Reward Distribution")
    ax.set_ylabel("Aggregate episode reward")
    ax.set_xticks([1, 2], labels)
    ax.text(
        0.02,
        0.98,
        "Easy: mean −0.201, 40% solved\nMedium: mean −0.983, 0.8% solved",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    save(fig, "qwen4_reward_distribution")


def thinking_efficiency() -> None:
    paths = [
        DATA_DIR
        / "qwen3.5-0.8b/qwen3.5-0.8b-thinking-easy-8rollouts.trajectories.json",
        DATA_DIR / "qwen3.5-2b/qwen3.5-2b-thinking-easy-8rollouts.trajectories.json",
        DATA_DIR
        / "qwen3.5-2b/qwen3.5-2b-thinking-easy-8rollouts-4096.trajectories.json",
        DATA_DIR / "qwen3.5-4b/qwen3.5-4b-thinking-smoke-4096.trajectories.json",
    ]
    labels = ["0.8B\n1,024", "2B\n1,024", "2B\n4,096", "4B\n4,096 smoke"]
    points = []
    for path in paths:
        data = load_json(path)
        tokens = []
        for episode in data["episodes"]:
            for step in episode["steps"]:
                usage = (step.get("response_metadata") or {}).get("usage") or {}
                value = usage.get("completion_tokens")
                if isinstance(value, int):
                    tokens.append(value)
        points.append(
            (
                sum(tokens) / len(tokens),
                data["summary"]["solved_rate"],
                data["summary"]["num_episodes"],
            )
        )

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    offsets = [(-5, 35), (15, 15), (10, 8), (10, 10)]
    alignments = ["right", "left", "left", "left"]
    for label, (mean_tokens, solved_rate, episodes), offset, alignment in zip(
        labels, points, offsets, alignments, strict=True
    ):
        ax.scatter(
            mean_tokens,
            solved_rate,
            s=80 + episodes * 1.5,
            color=COLORS["truncated"],
            edgecolor="#333333",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            f"{label}\nn={episodes}",
            (mean_tokens, solved_rate),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.7},
        )
    ax.set_title("Thinking Mode: Token Cost vs. Success")
    ax.set_xlabel("Mean completion tokens per model response (log scale)")
    ax.set_ylabel("Episode solved rate")
    ax.set_xscale("log")
    ax.set_ylim(-0.015, 0.20)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.text(
        0.02,
        0.95,
        "Bubble size indicates evaluated episodes.\nThinking runs are mostly malformed or truncated.",
        transform=ax.transAxes,
        va="top",
        color="#444444",
    )
    fig.tight_layout()
    save(fig, "thinking_efficiency")


def main() -> None:
    style()
    easy_nonthinking_comparison()
    qwen4_depth_profile()
    qwen4_reward_distribution()
    thinking_efficiency()
    print(f"Wrote graphs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
