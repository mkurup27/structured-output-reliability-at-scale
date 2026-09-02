#!/usr/bin/env python3
"""Figures for structured-output-reliability-at-scale.md.

All numbers are transcribed from the run summaries reported in the article, so
this script is a rendering step rather than an analysis step: it takes no input
files and reproduces exactly the tables in the text. Where a figure and a table
disagree, the table is authoritative and this file has a bug.

    python3 plot_figures.py               # render everything into ../figures/
    python3 plot_figures.py --only curve  # one figure by name

Needs matplotlib. Set MPLCONFIGDIR to a writable path if matplotlib complains
about its cache directory.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

INK = "#1a1d21"
MUTED = "#6b7078"
RULE = "#c9ccd1"
GRID = "#eceef1"

BLUE = "#0b6bcb"
TEAL = "#1f9c8f"
RED = "#d1495b"
GREY = "#8a8f98"
AMBER = "#b3742a"
SLATE = "#4a7fb5"


def _finish(ax, ylabel=None, xlabel=None):
    ax.grid(True, axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, labelpad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, labelpad=8)


def _titles(ax, title, subtitle, pad=30, y=1.042):
    ax.set_title(title, fontsize=13.5, fontweight="600", pad=pad, loc="left")
    ax.text(0, y, subtitle, transform=ax.transAxes, fontsize=8.4, color=MUTED)


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.normpath(path)}")


# --------------------------------------------------------------------------
# 1. Schema validity vs. concurrency, by arm.
# --------------------------------------------------------------------------
# The strict sustained series uses the --no-keepalive control value at c=400,
# since the pooled run's 0.9991 was a client connection-pool artifact rather
# than a schema failure. Plotting the pooled number would draw a droop that the
# control run showed does not exist.
def fig_curve():
    max_num_seqs = 128
    arms = [
        dict(label="vLLM strict, sustained (pinned xgrammar)",
             x=[1, 10, 50, 100, 200, 400], y=[1.0] * 6,
             color=BLUE, marker="o", lw=2.4, ls="-", z=5),
        # Coincides exactly with the sustained arm, so it is drawn over the top
        # with a dash pattern wide enough to let the line beneath show through.
        dict(label="vLLM strict, varying seed (T=0.7) — coincides at 1.00",
             x=[1, 10, 50, 100], y=[1.0] * 4,
             color=TEAL, marker="s", lw=2.0, ls=(0, (2.2, 3.4)), z=7),
        dict(label="vLLM prompt-only (no enforcement)",
             x=[1, 10, 50, 100], y=[0.67] * 4,
             color=GREY, marker="^", lw=2.0, ls="-", z=3),
        dict(label="DO Serverless, mistral-3-14B (managed)",
             x=[1, 10, 50, 100], y=[1.000, 0.925, 0.860, 0.8475],
             color=RED, marker="D", lw=2.4, ls="-", z=6),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=200)
    ax.axvline(max_num_seqs, color=RULE, lw=1.2, ls=":", zorder=1)
    ax.annotate(f"max_num_seqs = {max_num_seqs}\n(queueing starts here)",
                xy=(max_num_seqs, 0.632), xytext=(max_num_seqs * 1.13, 0.632),
                fontsize=8.2, color=MUTED, va="bottom", ha="left")

    for a in arms:
        ax.plot(a["x"], a["y"], color=a["color"], marker=a["marker"],
                markersize=6.0, linewidth=a["lw"], linestyle=a["ls"],
                label=a["label"], zorder=a["z"], markeredgecolor="white",
                markeredgewidth=0.9, clip_on=False)

    ax.annotate("holds at 1.00 with 266 requests queued",
                xy=(400, 1.00), xytext=(150, 0.945), fontsize=8.6, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9,
                                connectionstyle="arc3,rad=-0.18"))
    ax.annotate("degrades to ~0.85\n(all finish_reason: length)",
                xy=(100, 0.8475), xytext=(11.5, 0.795), fontsize=8.6, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.9,
                                connectionstyle="arc3,rad=0.16"))
    ax.annotate("a third of responses die at the parse rung\n(markdown code fence)",
                xy=(50, 0.67), xytext=(1.35, 0.705), fontsize=8.6, color=MUTED)

    ax.set_xscale("log")
    ax.set_xticks([1, 10, 50, 100, 200, 400])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.92, 480)
    ax.set_ylim(0.62, 1.025)
    ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    _finish(ax, "Schema validity rate", "Concurrency (log scale)")
    _titles(ax, "Schema validity vs. concurrency, by arm",
            "vLLM 0.27.1 / Qwen2.5-7B-Instruct / xgrammar 0.2.3 pinned / single H100 80GB"
            "  ·  managed arm is the mean of two runs")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.30), ncol=2, frameon=False,
              fontsize=9.0, handlelength=2.6, columnspacing=1.8, borderaxespad=0.0)
    _save(fig, "validity-vs-concurrency.png")


# --------------------------------------------------------------------------
# 2. What repair recovers, by document completion.
# --------------------------------------------------------------------------
# Bands are the uneven ones from the article's table (0-20, 20-30 ... 60-90,
# 90-100), plotted at band centres. The array-of-objects series is binned by
# decile instead, which is why it is drawn in its own panel.
def fig_repair():
    centres = [10, 25, 35, 45, 55, 75, 95]
    parses = [0.57, 0.67, 0.75, 0.87, 0.95, 1.00, 1.00]
    schema = [0.00, 0.02, 0.39, 0.76, 0.92, 1.00, 1.00]
    rules = [0.00, 0.01, 0.34, 0.73, 0.90, 1.00, 1.00]
    identical = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.23]

    arr_centres = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    arr_schema = [0.0] * 9 + [0.07]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.0), dpi=200,
                                  gridspec_kw=dict(width_ratios=[1.55, 1.0],
                                                   wspace=0.26))

    for y, label, color, marker, lw in [
        (parses, "Parses", GREY, "o", 1.8),
        (schema, "Schema-valid", SLATE, "s", 2.0),
        (rules, "Passes business rules", AMBER, "^", 2.0),
        (identical, "Byte-identical to the original", RED, "D", 2.8),
    ]:
        ax.plot(centres, y, color=color, marker=marker, markersize=5.6,
                linewidth=lw, label=label, markeredgecolor="white",
                markeredgewidth=0.9, clip_on=False)

    ax.axvspan(60, 90, color="#f6f7f9", zorder=0)
    ax.annotate("60–90%: every repaired document\npasses every automated check,\n"
                "and none of them is the\ndocument being written",
                xy=(74, 0.985), xytext=(52.5, 0.40), fontsize=8.4, color=INK,
                arrowprops=dict(arrowstyle="-", color=RULE, lw=0.9,
                                connectionstyle="arc3,rad=-0.22"))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    _finish(ax, "Rate", "Document completion at the cut point")
    _titles(ax, "What repair recovers: the flat schema",
            "4 schema families × 200 documents, 132,737 approximate-token cut points  ·  seed 20260813",
            pad=32, y=1.052)
    ax.legend(loc="upper left", frameon=False, fontsize=8.8, handlelength=2.4)

    ax2.plot(centres, schema, color=SLATE, marker="s", markersize=5.6, lw=2.0,
             label="Flat schema", markeredgecolor="white", markeredgewidth=0.9,
             clip_on=False)
    ax2.plot(arr_centres, arr_schema, color=TEAL, marker="o", markersize=5.6,
             lw=2.4, label="Array-of-objects (minItems + required)",
             markeredgecolor="white", markeredgewidth=0.9, clip_on=False)
    ax2.annotate("rejects a half-written\nrecord immediately:\n0.00 through nine deciles",
                 xy=(72, 0.012), xytext=(51, 0.30), fontsize=8.4, color=TEAL,
                 arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.9,
                                 connectionstyle="arc3,rad=-0.22"))
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.03, 1.06)
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax2.get_yaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    _finish(ax2, "Schema validity of the repaired document",
            "Document completion at the cut point")
    _titles(ax2, "The schema that fails loudest",
            "Schema validity only, by schema family",
            pad=32, y=1.052)
    ax2.legend(loc="upper left", frameon=False, fontsize=8.8, handlelength=2.4)

    _save(fig, "truncation-repair-by-completion.png")


# --------------------------------------------------------------------------
# 3. Latency moved, validity did not.
# --------------------------------------------------------------------------
def fig_sustained():
    labels = ["1", "10", "50", "100", "200", "400"]
    xs = list(range(len(labels)))
    p50 = [2.080, 1.944, 2.281, 2.830, 4.881, 9.386]
    p99 = [2.232, 2.500, 2.985, 3.884, 7.404, 11.279]
    queued = [0, 0, 0, 0, 72, 270]
    validity = [1.0] * 6

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.4, 6.6), dpi=200, sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.85, 1.0],
                                                   hspace=0.16))

    ax.plot(xs, p99, color=RED, marker="D", markersize=6.0, lw=2.4, label="p99",
            markeredgecolor="white", markeredgewidth=0.9, clip_on=False)
    ax.plot(xs, p50, color=SLATE, marker="o", markersize=6.0, lw=2.0, label="p50",
            markeredgecolor="white", markeredgewidth=0.9, clip_on=False)

    # The --no-keepalive control at c=400, same queue, client pool removed.
    ax.plot([5], [10.561], marker="D", markersize=8.0, markerfacecolor="white",
            markeredgecolor=RED, markeredgewidth=1.8, ls="none", zorder=6,
            label="p99, connection pooling disabled")
    ax.annotate("10.561 s with pooling disabled",
                xy=(5, 10.561), xytext=(2.75, 12.7), fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.9,
                                connectionstyle="arc3,rad=0.18"))
    ax.annotate("p99 grows 5.1× across the ramp",
                xy=(5, 11.279), xytext=(3.1, 9.0), fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="-", color=RULE, lw=0.9))
    ax.set_ylim(0, 15.0)
    _finish(ax, "End-to-end latency (s)")
    _titles(ax, "Contention moved p99 by roughly fivefold, and validity not at all",
            "Sustained 180-second windows per level, 30 s warm-up discarded  ·  "
            "vLLM 0.27.1 / Qwen2.5-7B-Instruct / max_num_seqs=128",
            pad=32, y=1.055)
    ax.legend(loc="upper left", frameon=False, fontsize=9.0, handlelength=2.4)

    ax2.bar(xs, queued, width=0.52, color="#dfe3e8", edgecolor=RULE, lw=0.8,
            zorder=2, label="Requests queued (max)")
    for x, q in zip(xs, queued):
        if q:
            ax2.text(x, q + 9, str(q), ha="center", fontsize=8.6, color=MUTED)
    ax2.set_ylim(0, 430)
    ax2.set_yticks([0, 100, 200, 300])
    _finish(ax2, "Queued (max)", "Concurrency")

    ax3 = ax2.twinx()
    ax3.plot(xs, validity, color=BLUE, marker="o", markersize=6.0, lw=2.6,
             zorder=4, label="Schema validity", markeredgecolor="white",
             markeredgewidth=0.9, clip_on=False)
    ax3.set_ylim(0.55, 1.06)
    ax3.set_yticks([0.6, 0.8, 1.0])
    ax3.get_yaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax3.set_ylabel("Schema validity", fontsize=10.5, labelpad=8, color=BLUE)
    ax3.tick_params(axis="y", colors=BLUE)
    for side in ("top", "left"):
        ax3.spines[side].set_visible(False)
    ax3.spines["right"].set_color(BLUE)
    ax3.spines["bottom"].set_color(RULE)
    ax3.annotate("flat at 1.00", xy=(2.0, 1.0), xytext=(2.0, 1.022),
                 fontsize=8.8, color=BLUE, ha="center")

    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels)
    handles = ax2.get_legend_handles_labels()[0] + ax3.get_legend_handles_labels()[0]
    labs = ax2.get_legend_handles_labels()[1] + ax3.get_legend_handles_labels()[1]
    ax2.legend(handles, labs, loc="lower left", bbox_to_anchor=(0.0, 0.03),
               frameon=False, fontsize=9.0, handlelength=2.4)

    _save(fig, "latency-vs-validity-sustained.png")


# --------------------------------------------------------------------------
# 4. Which keywords each backend actually enforces.
# --------------------------------------------------------------------------
ENFORCED, REJECTED, SILENT, NOVERDICT = 0, 1, 2, 3
STATE_STYLE = {
    ENFORCED: (TEAL, "✓", "Enforced"),
    REJECTED: (SLATE, "⊘", "Rejected up front (honest)"),
    SILENT: (RED, "✗", "Accepted, not enforced (silent)"),
    NOVERDICT: ("#b6bbc2", "?", "No clean verdict (truncation)"),
}


def fig_conformance():
    typed = [
        ("pattern", ENFORCED, ENFORCED),
        ("minLength", ENFORCED, ENFORCED),
        ("maxLength", ENFORCED, ENFORCED),
        ("format: email", ENFORCED, ENFORCED),
        ("format: unsupported", REJECTED, REJECTED),
        ("minimum", NOVERDICT, NOVERDICT),
        ("maximum", ENFORCED, ENFORCED),
        ("exclusiveMinimum", ENFORCED, ENFORCED),
        ("multipleOf", REJECTED, ENFORCED),
        ("minItems", ENFORCED, ENFORCED),
        ("maxItems", ENFORCED, ENFORCED),
        ("uniqueItems", REJECTED, REJECTED),
        ("patternProperties", REJECTED, ENFORCED),
        ("propertyNames", REJECTED, REJECTED),
        ("enum  (control)", ENFORCED, ENFORCED),
    ]
    untyped = [
        ("minimum, no sibling type", SILENT, ENFORCED),
        ("pattern, no sibling type", SILENT, ENFORCED),
        ("maxLength, no sibling type", SILENT, ENFORCED),
    ]
    rows = typed + untyped
    n = len(rows)
    x_left, x_right = -2.6, 4.5

    fig, ax = plt.subplots(figsize=(9.8, 8.4), dpi=200)
    for r, (name, xg, gd) in enumerate(rows):
        y = n - 1 - r
        for c, state in enumerate((xg, gd)):
            color, glyph, _ = STATE_STYLE[state]
            ax.add_patch(Rectangle((c + 0.06, y + 0.10), 0.88, 0.80,
                                   facecolor=color, edgecolor="white", lw=1.4,
                                   zorder=2))
            ax.text(c + 0.5, y + 0.50, glyph, ha="center", va="center",
                    fontsize=13, color="white", fontweight="600", zorder=3)
        untyped_row = SILENT in (xg, gd)
        ax.text(-0.14, y + 0.50, name, ha="right", va="center", fontsize=9.4,
                color=RED if untyped_row else INK,
                fontweight="600" if untyped_row else "normal", family="monospace")

    divider = len(untyped)
    ax.plot([-0.02, 2.02], [divider, divider], color=RULE, lw=1.2, ls=":")
    ax.annotate("every one of xgrammar's silent\nfailures is here: a fragment with\n"
                "no sibling \"type\" bypasses the\npreflight gate entirely",
                xy=(2.06, divider / 2), xytext=(2.28, divider / 2),
                fontsize=9.0, color=RED, va="center", ha="left")
    ax.annotate("", xy=(2.06, 0.15), xytext=(2.06, divider - 0.15),
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.6))

    for c, (backend, version) in enumerate([("xgrammar", "0.2.3"),
                                            ("guidance", "1.7.6")]):
        ax.text(c + 0.5, n + 0.52, backend, ha="center", fontsize=10.2,
                fontweight="600", color=INK)
        ax.text(c + 0.5, n + 0.20, version, ha="center", fontsize=8.6,
                color=MUTED)

    ax.text(x_left, -0.78,
            "xgrammar   9 enforced · 5 rejected up front · 3 silently ignored · 1 no verdict",
            fontsize=9.0, color=MUTED, family="monospace")
    ax.text(x_left, -1.24,
            "guidance  14 enforced · 3 rejected up front · 0 silently ignored · 1 no verdict",
            fontsize=9.0, color=MUTED, family="monospace")

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(-3.15, n + 1.05)
    ax.axis("off")
    ax.text(x_left, n + 1.62, "Which schema keywords each backend actually enforces",
            fontsize=13.5, fontweight="600", color=INK)
    ax.text(x_left, n + 1.26,
            "18 schemas × 5 trials at temperature 1.0, each prompting for a violating value  ·  "
            "vLLM 0.27.1, backend pinned, server restarted between runs",
            fontsize=8.4, color=MUTED)

    handles = [Rectangle((0, 0), 1, 1, facecolor=STATE_STYLE[s][0])
               for s in (ENFORCED, REJECTED, SILENT, NOVERDICT)]
    ax.legend(handles, [STATE_STYLE[s][2] for s in (ENFORCED, REJECTED, SILENT, NOVERDICT)],
              loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, frameon=False,
              fontsize=9.2, handlelength=1.5, columnspacing=1.6)

    _save(fig, "backend-conformance-grid.png")


# --------------------------------------------------------------------------
# 5. Enforcement relocated the failure rather than removing it.
# --------------------------------------------------------------------------
def fig_relocation():
    rungs = [
        ("Rung 1", "Termination metadata"),
        ("Rung 2", "Parse"),
        ("Rung 3", "Schema-validate"),
        ("Rung 4", "Semantic-validate"),
    ]
    # (caught_at_rung_index, note) per arm; None means nothing was caught.
    arms = [
        ("Prompt-only arm", "no enforcement", 1,
         "array_extract 50/50\nmarkdown code fence: json.loads\ndies on the leading backtick"),
        ("Strict arm", "xgrammar pinned", 3,
         "array_extract 50/50\ntotal 38.58 against line items\nsumming to 34.65"),
    ]

    fig, ax = plt.subplots(figsize=(10.6, 7.4), dpi=200)
    box_w, box_h, gap = 3.05, 0.82, 0.42
    col_x = {0: 0.0, 1: 5.15}
    top = len(rungs) * (box_h + gap)

    for ci, (title, sub, caught, note) in enumerate(arms):
        x = col_x[ci]
        cx = x + box_w / 2
        ax.text(cx, top + 0.52, title, ha="center", fontsize=11.4,
                fontweight="600", color=INK)
        ax.text(cx, top + 0.24, sub, ha="center", fontsize=9.0, color=MUTED)

        for ri, (num, name) in enumerate(rungs):
            y = (len(rungs) - 1 - ri) * (box_h + gap)
            hit = ri == caught
            ax.add_patch(FancyBboxPatch(
                (x, y), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.09",
                facecolor="#fdf0f2" if hit else "#f6f7f9",
                edgecolor=RED if hit else RULE, lw=2.0 if hit else 1.1, zorder=2))
            ax.text(x + 0.20, y + box_h / 2, num, ha="left", va="center",
                    fontsize=8.6, color=RED if hit else MUTED, fontweight="600")
            ax.text(x + 0.95, y + box_h / 2, name, ha="left", va="center",
                    fontsize=10.0, color=INK, fontweight="600" if hit else "normal")
            if hit:
                ax.text(x + box_w + 0.14, y + box_h / 2, "✗", ha="left",
                        va="center", fontsize=15, color=RED, fontweight="600")
            # Flow runs down the ladder, so the arrowhead sits on the lower box.
            if ri < len(rungs) - 1:
                ax.annotate("", xy=(cx, y - gap + 0.04), xytext=(cx, y - 0.04),
                            arrowprops=dict(arrowstyle="-|>", color=RULE, lw=1.2,
                                            mutation_scale=11))

        ax.text(cx, -0.34, note, ha="center", va="top", fontsize=8.6, color=RED)
        ax.text(cx, -1.62, "3 of 4 tasks usable", ha="center", fontsize=9.8,
                color=INK, fontweight="600")
        ax.text(cx, -1.90,
                "agent_multiturn, flat_extract, nested_toolcall\npass in both arms",
                ha="center", va="top", fontsize=8.3, color=MUTED)

    mid = (col_x[0] + box_w + col_x[1]) / 2
    ax.text(mid, top - 0.36, "the failure moved\ndown two rungs", fontsize=9.4,
            color=RED, fontweight="600", ha="center", va="center")
    ax.annotate("", xy=(col_x[1] - 0.13, 0.50), xytext=(col_x[0] + box_w + 0.46, 2.86),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8, mutation_scale=14,
                                connectionstyle="arc3,rad=-0.34"))

    ax.set_xlim(-0.45, 8.6)
    ax.set_ylim(-2.75, top + 1.55)
    ax.axis("off")
    ax.text(-0.45, top + 1.22,
            "Enforcement relocated the failure rather than removing it",
            fontsize=13.5, fontweight="600", color=INK)
    ax.text(-0.45, top + 0.94,
            "Four-task agent_mix set, identical prompts through both arms, concurrency 100, "
            "50 requests per task per arm",
            fontsize=8.4, color=MUTED)

    _save(fig, "failure-relocation-ladder.png")


# --------------------------------------------------------------------------
# 6. Schema cardinality is free; an undersized grammar cache is not.
# --------------------------------------------------------------------------
# Top panel reads TTFT p50 off the cardinality ladder at both cache settings.
# The 16-schema points come from the three-repeat onset runs rather than the
# ladder, which has no 16 rung at the default cache; the 1 MiB ladder's own 16
# rung ran 48 requests at concurrency 50 and is entirely burst onset, so it is
# excluded here for the same reason the article excludes it.
def fig_cardinality():
    labels = ["16", "64", "256", "1,024", "2,048"]
    xs = list(range(len(labels)))
    default = [0.481, 0.495, 0.497, 0.498, 0.489]
    reduced = [1.217, 1.348, 1.238, 1.231, 1.229]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.4, 7.6), dpi=200,
                                  gridspec_kw=dict(height_ratios=[1.3, 1.0],
                                                   hspace=0.62))

    ax.plot(xs, reduced, color=RED, marker="D", markersize=6.0, lw=2.4,
            label="1 MiB grammar cache", markeredgecolor="white",
            markeredgewidth=0.9, clip_on=False)
    ax.plot(xs, default, color=SLATE, marker="o", markersize=6.0, lw=2.2,
            label="Default cache (512 MiB)", markeredgecolor="white",
            markeredgewidth=0.9, clip_on=False)
    ax.annotate("flat across a 128× range in distinct grammars",
                xy=(2.0, 0.497), xytext=(1.15, 0.72), fontsize=8.6, color=INK,
                arrowprops=dict(arrowstyle="-", color=RULE, lw=0.9,
                                connectionstyle="arc3,rad=-0.2"))
    ax.annotate("penalty persists after the threshold is crossed",
                xy=(2.6, 1.235), xytext=(1.95, 1.60), fontsize=8.6, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.9,
                                connectionstyle="arc3,rad=0.2"))
    ax.set_ylim(0, 1.82)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    _finish(ax, "TTFT p50 (s)", "Distinct schemas offered")
    _titles(ax, "Cardinality costs nothing; undersizing the cache costs 2.5×",
            "Concurrency 50, streaming, three requests per schema  ·  "
            "vLLM 0.27.1 / xgrammar 0.2.3 / Qwen2.5-7B-Instruct",
            pad=32, y=1.06)
    ax.legend(loc="lower right", frameon=False, fontsize=9.0, handlelength=2.4)

    metrics = [
        ("Schema validity", 1.00, "1.0000 either way"),
        ("TTFT p50", 2.51, "0.489 s → 1.229 s"),
        ("Per-token decode", 2.08, "24.4 ms → 50.7 ms"),
        ("End-to-end p50", 2.16, "2.783 s → 6.007 s"),
        ("End-to-end p99", 2.23, "4.500 s → 10.057 s"),
        ("Seconds per completed\nrequest, sustained c=400", 3.64, "5,855 → 1,609 in 180 s"),
    ]
    ys = list(range(len(metrics)))[::-1]
    for y, (name, ratio, note) in zip(ys, metrics):
        colour = BLUE if ratio < 1.05 else RED
        ax2.barh(y, ratio, height=0.56, color=colour, alpha=0.85, zorder=3)
        ax2.text(ratio + 0.09, y, f"{ratio:.2f}×   {note}", va="center",
                 fontsize=8.5, color=MUTED)
    ax2.axvline(1.0, color=RULE, lw=1.0, zorder=2)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([m[0] for m in metrics], fontsize=9.0)
    ax2.set_xlim(0, 5.6)
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.grid(True, axis="x", color=GRID, lw=0.9, zorder=0)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_color(RULE)
    ax2.set_xlabel("Ratio, 1 MiB cache against default cache", fontsize=10.5,
                   labelpad=8)
    _titles(ax2, "Everything moved except the number teams watch",
            "2,048 distinct schemas, identical workload, server environment "
            "verified per restart with a return-to-default control",
            pad=26, y=1.10)

    _save(fig, "grammar-cache-cardinality.png")


FIGURES = {
    "curve": fig_curve,
    "repair": fig_repair,
    "sustained": fig_sustained,
    "conformance": fig_conformance,
    "relocation": fig_relocation,
    "cardinality": fig_cardinality,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(FIGURES), action="append",
                    help="render one figure by name; repeatable")
    args = ap.parse_args()
    for key in (args.only or sorted(FIGURES)):
        FIGURES[key]()
