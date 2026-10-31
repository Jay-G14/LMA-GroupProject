"""Report figures from the comparison files written by scripts/compare_systems.py.

  fig_macro_f1.{pdf,png}   claim-level macro-F1 with 95% bootstrap CI, every system, one series per eval set
  fig_cost.{pdf,png}       macro-F1 against LLM tokens per claim (log scale), one panel per eval set

Usage:
  python scripts/plot_results.py --comparisons to_transfer/dev/comparisons/dev.json \\
      to_transfer/open/comparisons/open.json --labels "SciFact dev" "SciFact-Open" --out reports/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms  # noqa: E402

ORDER = ["B0", "B1", "B1-9", "B1-SC", "B2", "M1", "M1-shared", "M1-aligned", "M1-untyped", "M1-LLMagg", "M2",
         "M2-random", "M2-claimiter"]
GROUP_ENDS = {"B2", "M1-LLMagg", "M2-claimiter"}       # hairline after each group: baselines | M1 | M2
COST_SYSTEMS = ["B0", "B1", "B1-9", "B1-SC", "B2", "M1", "M1-LLMagg", "M2", "M2-claimiter"]
# Amendment 5 re-scorings (post hoc) have their own table in the report, not a place in the frozen-results figure.
POST_HOC_SUFFIXES = ("-anch", "-anchsat", "-nosubject", "-relqual")


def post_hoc(system: str) -> bool:
    return system.endswith(POST_HOC_SUFFIXES) or system.startswith("B1-checked")
# Reference palette (dataviz skill, references/palette.md): categorical slots 1-2 pass every gate, all pairs,
# on this surface; series colors mark data only, text uses the ink tokens.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def macro_f1_figure(sets: list[dict], labels: list[str], out: Path) -> None:
    present = [s for s in ORDER if any(s in d for d in sets)]
    present += sorted({s for d in sets for s in d if not post_hoc(s)} - set(ORDER))   # e.g. M2-noconsol
    fig, ax = plt.subplots(figsize=(6.0, 0.3 * len(present) + 0.9), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    offsets = [0.0] if len(sets) == 1 else [-0.17, 0.17]
    for k, (d, label) in enumerate(zip(sets, labels)):
        ys, xs, lo, hi = [], [], [], []
        for i, s in enumerate(present):
            if s in d:
                ys.append(i + offsets[k])
                xs.append(d[s]["macro_f1"])
                lo.append(d[s]["macro_f1"] - d[s]["macro_f1_ci95"][0])
                hi.append(d[s]["macro_f1_ci95"][1] - d[s]["macro_f1"])
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="none", ecolor=SERIES[k], elinewidth=1.4, capsize=0, zorder=2)
        ax.plot(xs, ys, "o", color=SERIES[k], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.4,
                label=label, zorder=3)
    for i, s in enumerate(present[:-1]):
        if s in GROUP_ENDS:
            ax.axhline(i + 0.5, color=AXIS, linewidth=0.6)
    ax.set_yticks(range(len(present)), present, color=INK)
    ax.set_ylim(len(present) - 0.5, -0.5)
    ax.set_xlabel("claim-level macro-F1 (95% bootstrap CI)", color=INK_2, fontsize=8)
    if len(sets) > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(sets), fontsize=8, frameon=False,
                  labelcolor=INK_2, handletextpad=0.2, borderaxespad=0.2)
    save(fig, out, "fig_macro_f1")


def place_labels(ax, pts: list[tuple[float, float, str]]) -> None:
    """Name each dot without collisions: dots closer than a marker merge into one label, and each label takes
    the first candidate position (right, left, above, below, diagonals) clear of the dots and earlier labels."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    groups: list[list] = []
    for x, y, s in sorted(pts, key=lambda p: p[0]):
        px = to_px((x, y))
        for g in groups:
            if abs(g[0][0] - px[0]) < 9 and abs(g[0][1] - px[1]) < 9:
                g[1].append(s)
                break
        else:
            groups.append([px, [s], (x, y)])
    dots = [matplotlib.transforms.Bbox.from_bounds(g[0][0] - 5, g[0][1] - 5, 10, 10) for g in groups]
    taken = []
    candidates = [(7, 0, "left", "center"), (-7, 0, "right", "center"), (0, 7, "center", "bottom"),
                  (0, -7, "center", "top"), (6, 6, "left", "bottom"), (6, -6, "left", "top"),
                  (-6, 6, "right", "bottom"), (-6, -6, "right", "top")]
    for _, names, xy in sorted(groups, key=lambda g: -g[2][1]):
        for dx, dy, ha, va in candidates:
            t = ax.annotate(" / ".join(names), xy, xytext=(dx, dy), textcoords="offset points", fontsize=7,
                            color=INK_2, ha=ha, va=va)
            box = t.get_window_extent(renderer).expanded(1.05, 1.15)
            if (not any(box.overlaps(b) for b in taken + dots)
                    and ax.get_window_extent(renderer).contains(box.x0, box.y0)
                    and ax.get_window_extent(renderer).contains(box.x1, box.y1)):
                taken.append(box)
                break
            t.remove()
        else:   # nowhere clear: right of the dot anyway
            taken.append(ax.annotate(" / ".join(names), xy, xytext=(7, 0), textcoords="offset points", fontsize=7,
                                     color=INK_2, ha="left", va="center").get_window_extent(renderer))


def cost_figure(sets: list[dict], labels: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 2.9), facecolor=SURFACE, sharey=True,
                             squeeze=False)
    for k, (ax, d, label) in enumerate(zip(axes[0], sets, labels)):
        style(ax)
        pts = [(d[s]["tokens_per_claim"], d[s]["macro_f1"], s) for s in COST_SYSTEMS if s in d]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=SERIES[k], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v / 1000:g}K" if v >= 1000 else f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(label, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("LLM tokens per claim (log scale)", color=INK_2, fontsize=8)
        ax.set_xlim(min(p[0] for p in pts) / 1.8, max(p[0] for p in pts) * 2.2)
        ys = [p[1] for p in pts]
        ax.set_ylim(min(ys) - 0.03, max(ys) + 0.03)
    axes[0][0].set_ylabel("claim-level macro-F1", color=INK_2, fontsize=8)
    fig.tight_layout()
    for k, (ax, d) in enumerate(zip(axes[0], sets)):        # after layout, so label boxes are final
        place_labels(ax, [(d[s]["tokens_per_claim"], d[s]["macro_f1"], s) for s in COST_SYSTEMS if s in d])
    save(fig, out, "fig_cost")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparisons", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.labels) != len(args.comparisons) or len(args.comparisons) > len(SERIES):
        ap.error(f"one label per comparison file, at most {len(SERIES)} files")
    sets = [{t["system"]: t for t in json.loads(Path(p).read_text())["table"]} for p in args.comparisons]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    macro_f1_figure(sets, args.labels, out)
    cost_figure(sets, args.labels, out)
    print(f"wrote {out}/fig_macro_f1.{{pdf,png}} and {out}/fig_cost.{{pdf,png}}")


if __name__ == "__main__":
    main()
