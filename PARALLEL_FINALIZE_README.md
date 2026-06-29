# Parallel Program Finalization (MIGraphX) — Handoff README

**Purpose:** an opt-in change that finalizes a program's independent per-op work across
worker threads instead of one at a time — cutting the one-time finalization cost that
the first request pays (the dominant chunk of warm time-to-first-token).

**Quick links:** fork `https://github.com/aditya-dl/AMDMIGraphX` · branch
[`parallel-finalize`](https://github.com/aditya-dl/AMDMIGraphX/tree/parallel-finalize)
(also mirrored at `amd/dev/adilohia/parallel-finalize`) · base `develop` (`2b90a7914`) ·
**PR status: none (not opened).**

> ⚠️ **Status: built + validated, NOT merged.** Default-off, so it is a no-op until the
> env flag is set. Independent of the co-resident-cache / weight-sharing pair (different
> mechanism, different cost). Internal perf numbers/model names appear below for
> reference — **keep them out of any public PR description.**

---

## 0. Background (read first if you're new to this stack)

Skip this section if you already know MIGraphX finalize.

- **MIGraphX.** AMD's graph compiler/runtime. It compiles a neural-network model for a
  given input shape into a **program** (GPU kernels + execution context) and runs it.

- **Finalize.** Before a compiled program can run, it must be **finalized**: each
  operation gets its device-side setup done. The heaviest part is the program's many
  **code-object** ops — each one loads a compiled GPU module onto the device (a
  `hipModuleLoadData`-style call). A large model has hundreds of these.

- **Why it's slow and serial.** Historically `module::finalize` walks the ops and
  finalizes them **one at a time**. The code-object module loads are independent of each
  other, but they still run back-to-back, so their per-load latency adds up.

- **TTFT (time to first token).** Wall-clock time from "request received" to "first
  output token." Finalization is a one-time cost paid the first time a program runs;
  for an LLM it lands on the first token and is a visible part of TTFT.

- **op attribute.** In MIGraphX, every op can expose an `attributes()` map — a small set
  of flags the core can query generically (e.g. `pointwise`, `reduce`). This change adds
  one such flag so the core can parallelize finalize **without** knowing about any
  specific GPU op.

- **The idea in one sentence:** the independent module loads can be done in parallel
  across CPU threads, so finalization stops scaling with the number of code objects.

---

## 1. Original thesis

**Problem.** `module::finalize` finalizes ops serially. For an LLM the cost is dominated
by hundreds of code-object module loads that are mutually independent — there is no data
dependency forcing them to run one after another. Serial execution leaves that
parallelism on the table, and the cost is paid on the first request (TTFT).

**Hypothesis.** If the independent ops are finalized across worker threads, the per-load
latency overlaps and total finalization time drops substantially — with byte-identical
output, since only the *order/concurrency* of independent work changes, not the work
itself.

---

## 2. Implementation (what was built)

Two files. One env flag turns it on; default off is byte-identical to today's behavior.
The design keeps all GPU-specific knowledge in the GPU target — core stays
target-agnostic. (Diff: `module.cpp` +66, `code_object_op.hpp` +6.)

### `src/targets/gpu/include/migraphx/gpu/code_object_op.hpp` — the op opts in

```cpp
// "parallel_finalize" opts this op into module::finalize's parallel pass:
// its finalize() loads a GPU module and writes only its own instance (it does
// not touch the shared context), so finalizing many of them concurrently is safe.
value attributes() const { return {{"group", group()}, {"parallel_finalize", true}}; }
```

**What the code does:** the GPU code-object op declares a `parallel_finalize` attribute,
which is its promise that its `finalize()` only writes its own instance and never touches
shared state — i.e. it is safe to finalize many of them at once. This is how the GPU
side opts in *without* the core having to name a GPU op.

### `src/module.cpp` — core reads the attribute generically

```cpp
// An op may declare the "parallel_finalize" attribute to opt into being
// finalized concurrently ... Default unset = the original serial behavior below.
const auto can_parallel_finalize = [](instruction_ref ins) {
    return ins->module_inputs().empty() and
           ins->get_operator().attributes().get("parallel_finalize", false);
};
const std::size_t par_degree = value_of(MIGRAPHX_FINALIZE_PARALLEL{});
if(par_degree != 0 and not trace)
{
    std::vector<instruction_ref> parallel_ins;
    for(auto ins : iterator_for(*this))
        if(can_parallel_finalize(ins))
            parallel_ins.push_back(ins);
    if(not parallel_ins.empty())
    {
        const std::size_t n = parallel_ins.size();
        const std::size_t threads   = std::min(par_degree, n);
        const std::size_t min_grain = n / threads;
        par_for(n, min_grain, [&](auto i) {
            parallel_ins[i]->finalize(contexts[parallel_ins[i]->get_target_id()]);
        });
    }
    // Serial pass for the remaining instructions and submodules.
    for(auto ins : iterator_for(*this))
    {
        if(can_parallel_finalize(ins)) continue;
        ins->finalize(contexts[ins->get_target_id()]);
        for(const auto& smod : ins->module_inputs())
            smod->finalize(contexts);
    }
}
else
{
    // original serial loop (unchanged)
}
```

**What the code does:** when `MIGRAPHX_FINALIZE_PARALLEL` is set, the core collects every
op that declared `parallel_finalize` (and has no sub-modules), finalizes those across
worker threads via `par_for`, then runs a serial pass for everything else. The thread
count is `min(requested, number-of-such-ops)`, so it never over-spawns. Crucially, the
core never mentions `gpu::code_object` — it only queries a generic attribute, exactly
like existing passes query `pointwise`/`reduce`. When the flag is unset, control takes
the `else` branch, which is the original serial loop verbatim.

### The env flag

```cpp
// Finalizes ops that opt in via the "parallel_finalize" attribute concurrently.
// 0/unset = serial (original behavior); N > 0 = use up to N worker threads.
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FINALIZE_PARALLEL)
```

**What the code does:** declares the switch. Unset or `0` = original serial behavior;
`N > 0` = up to N worker threads, capped by the number of opt-in ops.

---

## 3. Verification done

Measured with the EP's no-warmup probe across the lead models; output checked
byte-identical. *(Internal numbers — keep out of any public PR.)*

Decode-program finalization, flag off vs on, ranked by absolute time saved:

| Model | kernel loads | finalize OFF | PARALLEL | abs. saved | ratio |
|---|---|---|---|---|---|
| Phi-3.5-mini (MHA) | 857 | 3,431.8 ms | 141.2 ms | ~3.3 s | 24x |
| DeepSeek-R1-Qwen-1.5B | 746 | 2,172.6 ms | 165.1 ms | ~2.0 s | 13.8x |
| Llama-3.2-1B | 434 | 1,307.9 ms | 228.0 ms | ~1.1 s | 5.7x |
| Qwen2.5-1.5B | 746 | 1,179.8 ms | 300.9 ms | ~0.9 s | 3.9x |

Lead with absolute seconds, not the ratio: Phi's "24x" is the biggest real win (~3.3 s),
larger than DeepSeek's "13.8x" (~2.0 s). Finalization collapses to ~140–300 ms
regardless of model — it stops scaling with the number of code objects.

| Check | Method | Result |
|---|---|---|
| Default-off = unchanged behavior | flag unset, output vs upstream | byte-identical (measured) |
| Output correctness with flag on | greedy token IDs, all 4 models | byte-identical to flag-off (measured) |
| Finalization speedup | probe, all 4 lead models | see table above (measured) |
| Thread-safety of the parallel pass | source review | code-object `finalize()` ignores the shared context and writes only its own instance — safe to run concurrently (reviewed) |

---

## 4. Status

- **Shippable:** yes, as an opt-in. Default-off path is byte-identical, so risk to
  existing users is minimal.
- **Payoff:** cuts the one-time finalization cost the first request pays — largest on
  models with the most code objects (Phi). No decode-time or VRAM cost; the change only
  reorders independent finalize work.
- **Independent of the co-resident-cache / weight-sharing pair** — different mechanism,
  different cost, different files. Can land on its own.
- **Not merged, no PR opened.** Branch is on `develop`, ready on the fork.

---

## 5. Known issues / risks

- **Layer (pre-empted).** Core `module::finalize` parallelizes via a **generic op
  attribute** (`parallel_finalize`), not by special-casing a GPU op name. Core has zero
  `gpu::`/`code_object` references. This deliberately avoids the "why does core know
  about a GPU op?" objection (a sibling PR, #5019, was rejected for putting GPU logic in
  core). The GPU specificity lives in `code_object_op.hpp`, the correct layer.
- **Thread-safety contract.** Correctness rests on the opt-in promise: an op that sets
  `parallel_finalize` must have a `finalize()` that touches only its own instance. The
  GPU code-object op satisfies this (verified by source read — it ignores its `context&`
  argument). Any future op that opts in must uphold the same contract. Ops with
  sub-modules are excluded from the parallel pass and stay serial.
- **Thread count.** `MIGRAPHX_FINALIZE_PARALLEL=N` uses up to N threads, capped by the
  number of opt-in ops. Very large N over-subscribes the CPU; a small fixed value (e.g.
  the hardware thread count) is the sensible setting. Default-off means none of this
  applies unless explicitly enabled.

---

## 6. Branches & artifacts

| Item | Value |
|---|---|
| Fork | `https://github.com/aditya-dl/AMDMIGraphX` |
| Branch | [`parallel-finalize`](https://github.com/aditya-dl/AMDMIGraphX/tree/parallel-finalize) |
| Mirror branch | `amd/dev/adilohia/parallel-finalize` (same code, no README) |
| Commit | `e6cdd4938` (code) |
| Base | `develop` (`2b90a7914`) — cherry-picked clean, no conflicts |
| Files | `src/module.cpp` (+66), `src/targets/gpu/include/migraphx/gpu/code_object_op.hpp` (+6) |
| Flag | `MIGRAPHX_FINALIZE_PARALLEL` — **default off** |
| PR status | none (not opened) |

---

## 7. How it relates to other tracks

- **Co-resident program cache (MIGraphX EP repo).** `ORT_MIGRAPHX_CORESIDENT_PROGRAMS`
  keeps both programs resident to avoid per-request reloads. Orthogonal: it *avoids* the
  reload on repeat requests; this change speeds up the finalize that the *first* reload
  pays. Different repo, different cost.
- **Weight-literal sharing (MIGraphX repo).** `MIGRAPHX_SHARE_LITERALS` dedupes weight
  VRAM across co-resident programs. Orthogonal to finalize timing; different file.
- **hipGraph decode-capture (PR #5019).** A different effort on the *decode* compute
  path (graph capture/replay). Orthogonal; different files. Note: it was rejected for a
  core-layer placement issue — this change was deliberately structured (generic
  attribute, no GPU op in core) to avoid the same objection.
- **Build-coupling rule.** MIGraphX statically links rocMLIR. Build this against **stock
  rocMLIR**, not the experimental M=1-GEMV branch (it carries an int4-compile bug).
