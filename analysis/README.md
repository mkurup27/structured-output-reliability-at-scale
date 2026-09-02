# analysis/ — turning runs into tables and figures

Three scripts. Two read result files and print the article's tables; the third draws the figures and reads nothing.

| Script | Input | Output |
|---|---|---|
| `summarize_runs.py` | `../results/*.json` | Per-arm tables, taxonomy breakdown, contention verdict |
| `compare_conformance.py` | `../results/conformance/*.json` | The keyword-enforcement grid |
| `plot_figures.py` | none | The six PNGs in `../figures/` |

## `summarize_runs.py`

```bash
python3 summarize_runs.py ../results
python3 summarize_runs.py --specimens ../results/r1_sustained_strict.json.gz
```

Takes one or more run files and emits, per arm: the configuration line (mode, taskset, load shape, token budget, temperature, seed), the per-level table (requests, schema validity, semantic validity, truncation, max queue depth, preemptions, KV-cache peak, p50, p99, cost per usable output), the taxonomy category and subcategory counts per level, and a derived contention verdict. `--specimens` prints the first captured raw output per failure category, deriving it from `records` when the aggregate specimen field is absent.

### The contention verdict

The summarizer reports which levels queued at all and which saw preemption, and flags any level where nothing queued as untested rather than as passing. That guard exists because its absence produced a wrong conclusion once: a ramp that stopped at concurrency 100 against a 128-slot batch showed flat validity and was read as evidence that concurrency does not affect validity, when it was a correct measurement of a server where queueing was impossible by construction.

It also reports offered against reached schema cardinality, so a run whose rotation fell short is not read as the cardinality on its label. The sustained reduced-cache arm is exactly that case: labelled 2,048, reached 1,609.

## `compare_conformance.py`

```bash
python3 compare_conformance.py ../results/conformance/conformance_xgrammar.json \
                               ../results/conformance/conformance_guidance.json
```

Takes one or more probe outputs and emits the enforcement grid, a resolved-version block per backend, a type-gate section showing which constraints stopped being enforced once `type` was omitted, and a cross-backend divergence list.

It checks the `enum` control case first and prints nothing but a warning if that case failed, because a backend that can't enforce `enum` isn't measuring the other seventeen keywords either. Passing all four of this study's probe files trips that guard, since `outlines` and `lm-format-enforcer` 500'd on every request. Pass the two working backends to reproduce the published grid.

`minimum` comes back as `partial data` on both backends. That's the honest verdict rather than a bug: the case is inconclusive, and the article reports it as such.

## `plot_figures.py`

```bash
python3 plot_figures.py                       # all six
python3 plot_figures.py --only curve          # one; repeatable
```

| `--only` | File |
|---|---|
| `curve` | `validity-vs-concurrency.png` |
| `repair` | `truncation-repair-by-completion.png` |
| `conformance` | `backend-conformance-grid.png` |
| `relocation` | `failure-relocation-ladder.png` |
| `sustained` | `latency-vs-validity-sustained.png` |
| `cardinality` | `grammar-cache-cardinality.png` |

Writes to `../figures/`, resolved relative to the script rather than the working directory, so it runs from anywhere. Needs `matplotlib`.

**This script renders values written into it.** It takes no input files: the per-arm numbers are transcribed from the runs into the source. So where a figure and a table disagree, the table is authoritative, and re-running this after new measurements means editing the values by hand. It will not pick them up from `../results/`.
