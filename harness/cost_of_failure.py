#!/usr/bin/env python3
"""
Experiment 3: cost per *valid* structured output.

Pure arithmetic over a published price. Nothing here is measured -- it is a model,
and the inputs are stated so you can substitute your own.

Price input: DigitalOcean Serverless Inference, Llama 3.3 Instruct-70B,
$0.65 / 1M input tokens and $0.65 / 1M output tokens.
Source: https://docs.digitalocean.com/products/inference/details/pricing/
(last verified on that page: 12 Aug 2026)
"""

IN_PRICE = 0.65 / 1_000_000   # $/token
OUT_PRICE = 0.65 / 1_000_000  # $/token

# A representative agent turn: a stable system prompt + tool definitions + history,
# emitting a structured tool call or final answer.
N_IN = 2_000
N_OUT = 400

BASE = N_IN * IN_PRICE + N_OUT * OUT_PRICE


def cost_full_regen(f, max_attempts=3):
    """Throw the bad output away, resend the original request unchanged."""
    total = 0.0
    valid = 0.0
    p_reach = 1.0
    for _ in range(max_attempts):
        total += p_reach * BASE
        valid += p_reach * (1 - f)
        p_reach *= f
    return total, valid


def cost_repair_retry(f, max_attempts=3):
    """Feed the invalid output plus the validator error back as new input."""
    total = 0.0
    valid = 0.0
    p_reach = 1.0
    for i in range(max_attempts):
        n_in = N_IN if i == 0 else N_IN + N_OUT + 80  # bad output + error message
        total += p_reach * (n_in * IN_PRICE + N_OUT * OUT_PRICE)
        valid += p_reach * (1 - f)
        p_reach *= f
    return total, valid


def table():
    print(f"Base cost of one attempt: ${BASE:.6f}  "
          f"({N_IN} in @ $0.65/1M, {N_OUT} out @ $0.65/1M)")
    print()
    hdr = (f"{'fail rate':>10} | {'regen $/valid':>14} | {'tax':>7} | "
           f"{'repair $/valid':>15} | {'tax':>7} | {'residual fail':>14}")
    print(hdr)
    print("-" * len(hdr))
    for f in (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15):
        tr, vr = cost_full_regen(f)
        tp, vp = cost_repair_retry(f)
        cr, cp = tr / vr, tp / vp
        print(f"{f*100:9.1f}% | ${cr:13.6f} | {(cr/BASE-1)*100:6.2f}% | "
              f"${cp:14.6f} | {(cp/BASE-1)*100:6.2f}% | {f**3*100:13.4f}%")


def retry_load_amplification():
    """Retries are not free capacity. They are additional offered load."""
    print()
    print("Offered load multiplier from retries (up to 3 attempts):")
    print(f"{'fail rate':>10} | {'requests per valid output':>26}")
    print("-" * 40)
    for f in (0.0, 0.01, 0.02, 0.04, 0.08, 0.15):
        req = 1 + f + f * f
        valid = 1 - f ** 3
        print(f"{f*100:9.1f}% | {req/valid:26.4f}")


def budget_failure_is_not_independent():
    """
    The geometric model above assumes attempts are independent. Truncation
    failures are not: if the request needs more tokens than the budget allows,
    every retry at the same budget fails the same way.
    """
    print()
    print("Expected attempts to first success, 1 - f independent per attempt:")
    for f in (0.02, 0.04, 0.08):
        print(f"  f={f:.2f} -> {1/(1-f):.3f} attempts")
    print("  truncation at a fixed budget -> attempts diverge (p_success = 0)")


def grammar_cache_cardinality():
    """
    vLLM's xgrammar compiler cache: VLLM_XGRAMMAR_CACHE_MB, default 512.
    vllm/envs.py comment says "512 MB should be enough for roughly 1000 JSON
    schemas," but the code multiplies by 1024**2, so the cache is 512 MiB in
    practice and the per-schema figure below is a rule of thumb from a source
    comment, not a hard ceiling: the cache is bounded by bytes, not schema
    count, and compiled grammar size varies with schema complexity. Report
    "above the estimate," not a confident "will thrash."
    """
    print()
    cache_mib = 512
    schemas = 1000
    print(f"vLLM xgrammar cache default: {cache_mib} MiB ~ {schemas} schemas "
          f"(~{cache_mib*1024/schemas:.0f} KiB/schema, rule of thumb only)")
    print(f"{'tools':>8} | {'variants/tool':>14} | {'distinct schemas':>17} | {'vs. ~1,000-schema estimate':>27}")
    print("-" * 78)
    for tools, variants in ((12, 1), (40, 1), (40, 4), (120, 4), (300, 4)):
        n = tools * variants
        if n <= schemas * 0.05:
            verdict = "far below"
        elif n <= schemas * 0.2:
            verdict = "below"
        elif n <= schemas:
            verdict = "approaching"
        else:
            verdict = "above the estimate"
        print(f"{tools:8d} | {variants:14d} | {n:17d} | {verdict:>27}")


if __name__ == "__main__":
    table()
    retry_load_amplification()
    budget_failure_is_not_independent()
    grammar_cache_cardinality()