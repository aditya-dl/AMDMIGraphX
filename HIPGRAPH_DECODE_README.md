# hipGraph (GPU graph capture) for LLM decode — track README (handoff)

**Purpose of this doc:** hand off the hipGraph decode-capture work, PR
**[ROCm/AMDMIGraphX#5019](https://github.com/ROCm/AMDMIGraphX/pull/5019)**, including the maintainer
review that lands on it. Single reference for whoever takes this over while I'm out. NOT in the PR
branch — lives on the internal
[`amd/dev/adilohia/hipgraph-decode-capture`](https://github.com/aditya-dl/AMDMIGraphX/tree/amd/dev/adilohia/hipgraph-decode-capture)
branch so it never reaches upstream reviewers.

**Fork:** https://github.com/aditya-dl/AMDMIGraphX · **PR:**
[#5019](https://github.com/ROCm/AMDMIGraphX/pull/5019) · **superseding PR:**
[#4956](https://github.com/ROCm/AMDMIGraphX/pull/4956) · full branch/commit table in §6.

> ⚠️ **Read §5 first if you're triaging the PR.** A maintainer (pfultz2) has indicated our approach is
> superseded by an in-flight implementation (#4956). The mechanism below works and is validated, but
> the *placement* (`program::eval`) is the thing under review.

---

## 0. Background (read this first if you're new to the area)

Plain-language orientation so the rest is readable without deep GPU/MIGraphX knowledge. Skip if you
already work in this stack.

- **MIGraphX** is AMD's inference engine: it takes a model, compiles it into a list of GPU operations,
  and runs them. It's the layer being changed here.
- **LLM inference has two phases.** *Prefill* processes the prompt once; *decode* then generates the
  answer one token (word-piece) at a time, re-running the model per token. Decode dominates latency
  for chat-style use, so it's where we optimize. **"per-token" = once per generated token.**
- **The problem we attacked — dispatch overhead.** To run the model for one token, the engine tells
  the GPU to run ~50+ small operations ("kernels"), issuing them **one at a time** from the CPU. Each
  hand-off has CPU cost, and the gaps between them let the GPU clock throttle down. On fast GPUs this
  per-launch overhead — not the actual math — is a big chunk of per-token time.
- **hipGraph (the fix) — this is GPU "graph capture."** If you already know graph capture (NVIDIA
  CUDA Graphs / DirectML command-list replay), that's exactly what this is; hipGraph is HIP/AMD's
  version. For everyone else: it lets you **record** a sequence of GPU operations once and **replay**
  the whole sequence with a single command, so instead of ~50 hand-offs per token you do one. This is
  "capture/replay."
- **fp16 vs int4 ("quantization").** Model weights can be stored at full precision (**fp16**, 16-bit)
  or compressed to **int4** (4-bit) to save memory/bandwidth. int4 needs an extra on-the-fly
  "unpack/dequantize" step. Our change **only** turns on hipGraph for fp16 models — we measured that
  it makes int4 *slower*, so int4 is deliberately left on the normal path (the "gate," §2d).
- **The review situation.** This work is an open pull request (#5019). A senior maintainer has said
  the *idea* is fine but it's built in the wrong place, and there's already another in-progress
  version (#4956) he prefers. So the open question isn't "does it work" (it does) — it's "which
  implementation lands, and does our int4 safeguard get carried over." See §5.
- **A few names you'll see:** *develop* = the upstream main branch the PR targets; *rel-2608* = the
  internal release branch our other work is built on; *EP* = "execution provider," the plug-in that
  lets the inference runtime use MIGraphX; *pass* / *op* = MIGraphX's two normal ways to add
  functionality (a compile-time transform, and a runtime operation) — the maintainer wants the
  feature expressed as those rather than by editing core engine code.

---

## 1. Original thesis

LLM decode is bottlenecked on discrete GPUs by **host dispatch overhead**, not kernel compute.
`program::eval` issues the per-token kernel sequence one launch at a time (`generic_eval` → one
`hipExtModuleLaunchKernel` per op, ~50+ launches/token). The per-launch host cost plus the GPU-clock
throttle from the resulting dispatch bubbles is a measurable fraction of per-token latency. Other
backends already avoid this (CUDA Graphs; DirectML D3D12 command-list replay). **Thesis: capture the
decode kernel loop into a hipGraph once and replay it with one launch per token.**

## 2. Implementation (what was built)

Commit `f8b7c6c17` — 4 files, +198/−2. The mechanism has three parts: **capture/replay primitives**
on the GPU context, **routing** through eval, and a **fp16-only gate**.

### 2a. GPU context — capture/replay primitives
`src/targets/gpu/include/migraphx/gpu/context.hpp`

RAII handles + the capture/replay entry point. `execute()` is the single entry: off → run eagerly;
graph already built → replay; first eval → capture, instantiate, replay.

```cpp
using hip_graph_ptr      = MIGRAPHX_MANAGE_PTR(hipGraph_t, hipGraphDestroy);
using hip_graph_exec_ptr = MIGRAPHX_MANAGE_PTR(hipGraphExec_t, hipGraphExecDestroy);

bool is_graph_enabled() const   // opt-in, not cross-compiling, AND capturable (see gate)
{ return enabled(MIGRAPHX_ENABLE_HIPGRAPH{}) and not is_cross_compile() and graph_capturable; }

void execute(const std::function<void()>& run_kernels)
{
    if(not is_graph_enabled()) { run_kernels(); return; }   // eager (default)
    if(has_graph())            { replay_graph(); return; }   // steady-state: 1 launch
    begin_graph_capture();                                    // first eval: capture...
    run_kernels();
    if(end_graph_capture()) replay_graph();                   // ...instantiate + run once
    else                    run_kernels();                    // capture failed -> eager
}
```
**What the code does, line by line:**
- The two `using` lines wrap the raw HIP graph handles (`hipGraph_t`, `hipGraphExec_t`) in
  smart-pointer types that auto-destroy them — so we never leak a captured graph.
- `is_graph_enabled()` is the master on/off check, true only when **all three** hold: the user set the
  env flag, we're not cross-compiling (compiling for a GPU that isn't present), and this program is
  capturable (the fp16 gate — §2d). If any is false, hipGraph is skipped entirely.
- `execute()` is the one function the rest of the engine calls to run the kernel loop. It has three
  cases: (1) feature off → just run the kernels normally ("eager"); (2) a graph was already captured
  on a previous token → replay it with one call and return; (3) first token → start capture, run the
  loop once (which *records* the kernels rather than running them), finalize the graph, and replay it
  once to actually produce this token. If capture fails for any reason, fall back to running eagerly so
  the result is always correct.

`begin/end_graph_capture()` wrap `hipStreamBeginCapture` (ThreadLocal) … `hipStreamEndCapture` +
`hipGraphInstantiate`; `replay_graph()` is `hipGraphLaunch`. State held on the context:
`captured_graph`, `graph_exec`, and `bool graph_capturable = true` (set false by the gate).

### 2b. program::eval — routing  ⚠️ THIS is the contested change
`src/program.cpp` (the single-context branch)

```cpp
else if(contexts.size() == 1)
{
    contexts.front().execute([&] {
        ret = generic_eval(*this, contexts, params, [&](auto&&, auto f) { return f(); });
        impl->graph_cached_results = ret;          // captured eval produces no output; cache it
    });
    if(ret.empty())
        ret = impl->graph_cached_results;          // replay path: reuse cached output args
}
```
**What the code does:** `program::eval` is the engine's run function. The `contexts.size() == 1`
branch is the common single-GPU case (which decode hits). Instead of running the kernel loop directly,
it now hands the loop to `context::execute()` (from §2a) as a callback — so the context can decide to
run it eagerly *or* capture/replay it. The callback runs `generic_eval` (the actual per-op loop) and
stashes its result in `graph_cached_results`. After `execute()` returns, if `ret` is empty (which
happens on a replay, because replaying a captured graph doesn't go through the callback) we substitute
the cached result.

Why the cache: under `hipStreamBeginCapture` the launches are *recorded, not executed*, so the
capture eval returns empty; on replay we return the cached output arguments. Valid because static-
shape decode reuses fixed device buffers. Flag off → `execute()` just runs the loop → byte-identical
to the prior path. **This core-file change is what the maintainer objects to (§5).**

### 2c. Type-erased context — the hook
`src/include/migraphx/context.hpp`

Adds an `execute(run_kernels)` method to the type-erased `context` whose **default just runs the
loop** (`execute_context` free function), so non-GPU targets are unaffected; the GPU context overrides
it with 2a. (This adds a method to a widely-implemented interface — also part of what review flags.)

### 2d. fp16-only gate
`src/targets/gpu/fuse_mlir.cpp` + the `graph_capturable` flag in 2a

hipGraph capture **regresses int4/fp4 decode (measured up to ~2× slower on discrete GPUs)**. The
mechanism behind the regression was **not root-caused** — it was measured (interleaved A/B on navi31
/ STX-Halo, quantized vs fp16) and gated on empirically. So this is a measured-fact gate, not a
mechanistic one; if the gate is ported elsewhere, the regression should be re-confirmed rather than
assumed from a theory. `fuse_mlir::apply` scans the module (pre-lowering, while op names are intact)
and marks the context non-capturable if any quantized/low-bit op is present:

```cpp
static const std::array<std::string,4> low_bit_ops =
    {{"unpack_int4","unpack_fp4","dequantizelinear","quant_dot"}};
for(const auto& ins : mpm.get_module())
    if(contains(low_bit_ops, ins.name())) { ctx->set_graph_not_capturable(); break; }
```
**What the code does:** during compilation, `fuse_mlir::apply` walks every instruction in the model
(`mpm.get_module()`). If it finds any op whose name is one of the four quantization/low-bit markers
(`unpack_int4`/`unpack_fp4`/`dequantizelinear`/`quant_dot`), it flips the context's `graph_capturable`
flag to false and stops scanning. That flag is the third condition in `is_graph_enabled()` (§2a), so a
quantized model can never enter the capture path — it always runs eagerly. A pure fp16 model contains
none of those ops, so the flag stays true and capture is allowed.

Allowlist-by-absence → every int4 variant (AWQ/RTN, block-32/128) + int8/fp4 stays eager; only fp16
(none of these ops) captures. `is_graph_enabled()` enforces it. Cheap pre-lowering scan; no per-token
or compile cost, none when the feature is off. **This gate is the part of the work most likely to be
additive even if the capture mechanism is replaced (§5).**

### Data flow (end to end)
1. compile: `fuse_mlir` sets `graph_capturable` (false if quantized) on the context.
2. first decode eval: `is_graph_enabled()` true (fp16+flag) → capture loop → instantiate → cache
   output args → replay once.
3. subsequent evals: `has_graph()` → single `hipGraphLaunch`, host-side loop skipped.
4. flag off / quantized / cross-compile / capture failure → eager loop, byte-identical to baseline.

## 3. Verification done

| Check | Result |
|---|---|
| Greedy decode output, off vs on (fp16) | byte-identical |
| Gate: int4 program, flag on | marked non-capturable → eager (no regression); confirmed via a temporary capture-engaged stderr marker (since removed) |
| Gate: fp16 program, flag on | captures + replays |
| Builds on the PR base (`develop`) | clean full-target build, exit 0 |
| Default off | byte-identical to prior path |
| New external dependencies | none |

**Measured fp16 steady-state decode throughput, off→on, interleaved same-session A/B, md5-verified
builds** (internal numbers — keep OUT of the public PR; generalized ranges only there):

| Platform | Off→On |
|---|---|
| RX 9070 XT (gfx1201, RDNA4) | Llama-1B ~+24%; DeepSeek-1.5B / Qwen-1.5B ~+18–20% |
| RDNA3 dGPU (navi31) | all four lead models ~+6 to +16% |
| Strix Halo APU (gfx1150) | all four ~+2–3% (memory-bound → less dispatch headroom; expected) |

Win scales with how dispatch-bound the platform is (largest on the highest-bandwidth dGPU).

## 4. Architecture note (why the placement is contested)

The implementation puts capture/replay in **`program::eval`** (core, target-agnostic) + a hook on the
**type-erased context** — for a **GPU-only** feature. It works, but it's the wrong *layer*: capture is
GPU-specific and shouldn't live in core that every target (CPU/ref/gpu) runs through. The MIGraphX-
idiomatic way to express "rewrite the program for the GPU" is a **pass** + a **custom op** (e.g.
`mlss_conv` in this same tree is capture-as-op). See the learning note
`docs/learning-notes/ask-layer-question-before-touching-core.md` in the optimization repo for the full
analysis.

## 5. Maintainer review — current status (READ THIS)

pfultz2 on PR #5019: *"there is no reason to modify `program::eval` as you can just write an op to do
the execution. … see #4956 which implements hip graph and it handles when the pointer change."*

**[#4956 "Add support for HipGraph"](https://github.com/ROCm/AMDMIGraphX/pull/4956)** (pfultz2, DRAFT,
~1483 additions) does the same feature with **zero core/eval changes**:
- a GPU pass **`hipgraphify`** that partitions the module into maximal runs of capturable
  instructions and rewrites each into…
- a **`hip::graph` op** (`hip_graph.cpp`) holding the captured `hipGraphExec`, with an
  **`exec::update()`** that re-points the graph when buffer pointers change (the invariant we worked
  around by caching output args + assuming fixed buffers).

**Implication:** the capture *mechanism* in #5019 is likely superseded by #4956. The PR is unlikely to
merge as-structured. Realistic outcomes:
1. Close #5019 in favor of #4956; OR
2. Contribute #5019's **distinct value — the fp16/int4-gate (§2d)** — onto #4956 (open question:
   does #4956's `is_capturable` predicate already exclude quantized runs? needs checking before
   claiming the gate is additive); OR
3. Rework #5019 to the pass+op pattern (large, duplicates #4956 — not recommended).

**What's left / next actions for the merge owner:**
1. Get the timeline + intent for #4956 from pfultz2 (a Teams message was being drafted: ask ETA,
   whether to close #5019, and whether any piece is worth keeping separate).
2. Verify whether #4956 already handles the quantized-regression case (read its `is_capturable`).
   If not, the §2d gate is the thing to land — onto #4956, not as #5019.
3. Upstream CI on #5019 only matters if #5019 stays alive; deprioritize until (1).

## 6. Branches & artifacts

| Item | Value |
|---|---|
| **PR (ours)** | https://github.com/ROCm/AMDMIGraphX/pull/5019 (base `develop`) |
| Superseding PR | https://github.com/ROCm/AMDMIGraphX/pull/4956 (pfultz2, draft) |
| Fork | `aditya-dl/AMDMIGraphX` — https://github.com/aditya-dl/AMDMIGraphX |
| PR branch (fork) | [`amd/dev/adilohia/hipgraph-decode-capture-develop`](https://github.com/aditya-dl/AMDMIGraphX/tree/amd/dev/adilohia/hipgraph-decode-capture-develop) — commit `0647ae162`, 1 commit on `develop`, includes CHANGELOG. This is what PR #5019 is opened from. |
| Internal branch (this README) | [`amd/dev/adilohia/hipgraph-decode-capture`](https://github.com/aditya-dl/AMDMIGraphX/tree/amd/dev/adilohia/hipgraph-decode-capture) — rel-2608 base; holds this README + the code; NOT PR'd. README at [`HIPGRAPH_DECODE_README.md`](https://github.com/aditya-dl/AMDMIGraphX/blob/amd/dev/adilohia/hipgraph-decode-capture/HIPGRAPH_DECODE_README.md) (visible once the latest commit is pushed). |
| Enable flag | `MIGRAPHX_ENABLE_HIPGRAPH=1` (default off) |
| Code commit | `f8b7c6c17` (on both branches) |

## 7. How to reproduce the A/B (internal env)

Same dll for both arms; only the env var differs. Build vs **stock** rocMLIR (the GEMV rocMLIR breaks
int4 compile — unrelated track). Run an fp16 decode model flag-off then flag-on, interleaved; compare
steady-state decode throughput. Quantized models should show ~no change (gated → eager). Output must
be byte-identical off vs on.
