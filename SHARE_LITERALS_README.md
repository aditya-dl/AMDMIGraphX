# Shared Weight Literals (MIGraphX GPU target) — Handoff README

**Purpose:** an opt-in, content-addressed pool that lets multiple compiled programs on
the same device share a single VRAM copy of byte-identical weight tensors — so keeping
an LLM's prefill and decode programs resident at once does not double weight VRAM.

**Quick links:** fork `https://github.com/aditya-dl/AMDMIGraphX` · branch
[`share-literals`](https://github.com/aditya-dl/AMDMIGraphX/tree/share-literals)
(also mirrored at `amd/dev/adilohia/share-literals`) · base `develop` (`2b90a7914`) ·
**PR status: none (not opened).**

> ⚠️ **Status: built + validated, NOT merged.** Default-off, so it is a no-op until the
> env flag is set. This is the VRAM half of a pair: it pays off when the MIGraphX EP's
> co-resident program cache is also enabled (see §5, §7). Internal perf numbers/model
> names appear below for reference — **keep them out of any public PR description.**

---

## 0. Background (read first if you're new to this stack)

Skip this section if you already know MIGraphX literals and prefill/decode.

- **MIGraphX.** AMD's graph compiler/runtime. It takes a neural-network model, compiles
  it for a specific input shape into a **program** (GPU kernels + execution context),
  and runs it.

- **Literal / weight.** A **literal** is a constant tensor baked into the model — for an
  LLM, the literals are the **weights** (the trained parameters). They are by far the
  largest thing in VRAM (gigabytes for a multi-billion-parameter model).

- **`gpu_literal` and `finalize`.** Inside MIGraphX's GPU target, each weight is
  represented by a `gpu_literal` op. When a program is made ready to run (**finalize**),
  each `gpu_literal` uploads its bytes from host memory to the GPU (`to_gpu`) and holds
  onto that device buffer. One upload = one VRAM copy of that weight.

- **Prefill vs decode (LLM generation).** A language model answers in two phases —
  prefill (read the whole prompt) and decode (emit one token at a time). They have
  different input shapes, so MIGraphX compiles them into **two separate programs**. Each
  program has its *own* set of `gpu_literal`s → if both programs are kept resident, the
  **same weights get uploaded twice** = ~2× weight VRAM.

- **Why this matters now.** A companion change in the MIGraphX EP keeps both the prefill
  and decode programs resident at once (to avoid reloading them on every request). That
  is great for latency but, without this change, it doubles weight VRAM — which can put
  large models over the memory budget. This change removes that doubling.

- **Content-addressed / dedup.** "Content-addressed" means we key data by a hash of its
  *bytes*, not by a name. If two weights have identical bytes, they hash to the same key
  and can share one buffer. We always byte-compare before sharing, so a hash collision
  can never alias two genuinely different weights.

---

## 1. Original thesis

**Problem.** Keeping two programs (prefill + decode) resident means each holds its own
device copy of the weights. The weights are identical between the two programs, so this
is a pure ~2× VRAM waste — and it's the blocker that would otherwise make the EP's
co-resident program cache too memory-hungry for large models.

**Hypothesis.** If `gpu_literal`s consult a shared, process-wide pool keyed by weight
content, the second (and later) program that finalizes an identical weight can alias the
*existing* device buffer instead of uploading a new one — bringing co-resident VRAM back
to roughly single-program levels, with no change to numerical output.

---

## 2. Implementation (what was built)

One file changed: `src/targets/gpu/write_literals.cpp`. One env flag turns it on;
default off is byte-identical to today's behavior. (Diff: +88 / -1.)

### The env flag

```cpp
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SHARE_LITERALS)
```

**What the code does:** declares the on/off switch `MIGRAPHX_SHARE_LITERALS`. Unset
(default) = the original per-program upload behavior.

### The content hash + collision guard

```cpp
// FNV-1a 64-bit over the literal's raw host bytes, used as the content key. The
// hash only selects a candidate; the bytes are always compared before sharing.
std::uint64_t literal_content_hash(const argument& data) { /* FNV-1a over bytes */ }

// Exact equality of two host literals (same byte length and identical bytes).
// Guards against FNV-1a hash collisions: two distinct weights that hash equal
// must not be aliased to the same device buffer.
bool same_bytes(const argument& a, const argument& b)
{
    const std::size_t n = a.get_shape().bytes();
    if(n != b.get_shape().bytes())
        return false;
    return std::memcmp(a.data(), b.data(), n) == 0;
}
```

**What the code does:** `literal_content_hash` produces a 64-bit fingerprint of a
weight's bytes — a fast way to *find a candidate* match in the pool. `same_bytes` is the
safety net: before two weights are ever shared, their full bytes are compared. The hash
narrows the search; the byte-compare guarantees correctness. So even if two different
weights happened to hash to the same value, they would fail `same_bytes` and **not** be
shared.

### The pool

```cpp
// Process-lifetime, per-(device,content-hash) pool of already-uploaded weights.
// Each entry retains the host literal (for the collision byte-compare) and the
// device buffer; gpu_literals that opt in share the SAME device `argument`
// (shared_ptr-refcounted), so N identical weights => 1 VRAM copy.
struct pooled_literal
{
    argument host; // host bytes, for the collision check
    argument gpu;  // device buffer, shared on a match
};
struct shared_literal_pool { std::mutex mtx; std::unordered_map<std::string, pooled_literal> buffers; ... };
```

**What the code does:** a single process-wide table mapping a key (device name + content
hash) → an already-uploaded weight. Each entry keeps both the host bytes (needed for the
collision byte-compare) and the GPU buffer (the thing that gets shared). A `mutex`
guards it because finalize can run on several compile threads at once. A MIGraphX
`argument` is shared-pointer-backed, so handing out the same `gpu` buffer to several
programs means one VRAM allocation with a bumped reference count.

### Using the pool in `gpu_literal::finalize`

```cpp
if(enabled(MIGRAPHX_SHARE_LITERALS{}) and not host)
{
    const std::string key = ctx.get_current_device().get_gfx_name() + ":" +
                            std::to_string(literal_content_hash(data));
    auto& pool = shared_literal_pool::instance();
    std::lock_guard<std::mutex> lock(pool.mtx);
    auto it = pool.buffers.find(key);
    if(it != pool.buffers.end() and same_bytes(it->second.host, data))
    {
        gpu_data = it->second.gpu.share(); // alias existing VRAM buffer (refcount++)
        return;
    }
    // Miss, or a hash collision with different bytes: upload a private copy.
    gpu_data = to_gpu(data);
    if(it == pool.buffers.end())
        pool.buffers.emplace(key, pooled_literal{data.share(), gpu_data.share()});
    return;
}
if(host) gpu_data = register_on_gpu(data);
else     gpu_data = to_gpu(data);
```

**What the code does:** when sharing is enabled and the weight lives in device VRAM
(`not host`), it builds a key from the device name + content hash and looks in the pool.
On a **hit that passes the byte-compare**, it aliases the existing device buffer (no new
upload — this is where the VRAM saving happens) and returns. On a **miss**, it uploads
the weight normally and registers it so a later identical weight can share it. On a
**hash collision with different bytes**, it safely falls back to a private upload (does
not overwrite the pool entry). If the flag is off — or the literal is host-pinned, not
device VRAM — control falls through to the original `register_on_gpu` / `to_gpu` path,
unchanged.

---

## 3. Verification done

On a discrete RDNA-class GPU, measured with the EP's no-warmup probe + `hipMemGetInfo`,
with the companion co-resident program cache enabled (that's the scenario this change
exists for). *(Internal numbers — keep out of any public PR.)*

| Check | Method | Result |
|---|---|---|
| Default-off = unchanged behavior | flag unset, output + VRAM vs upstream | byte-identical, no VRAM change (measured) |
| Sharing reclaims the co-residency VRAM | both flags on, `hipMemGetInfo` (DeepSeek-1.5B) | co-resident-only ~7,056 MB → with sharing **~3,613 MB (≈ baseline ~3,640 MB)** (measured) |
| Output correctness with flag on | greedy token IDs, every iteration | byte-identical to flag-off (measured) |
| Collision safety | code review | `same_bytes` byte-compare gates every share; collision falls back to private copy — reviewed |
| Behavior under many distinct models | — | **not tested** (retention concern, see §5) |

---

## 4. Status

- **Shippable:** yes, as an opt-in. Default-off path is byte-identical, so risk to
  existing users is minimal.
- **Payoff (honest):** this change has **no standalone latency benefit** — its entire
  value is reclaiming the VRAM that the companion co-resident program cache would
  otherwise double. Measured: brings co-resident weight VRAM back to ≈ single-program
  baseline. Ship it **with** the EP cache, not alone.
- **Not merged, no PR opened.** Branch is on `develop`, ready on the fork.

---

## 5. Known issues / risks

- **Process-lifetime retention (the main reviewer concern).** The pool holds a strong
  reference to every uploaded weight for the life of the process. This is intentional
  and correct for the resident-LLM use case (the weights stay live as long as the
  resident programs do). But a process that repeatedly **loads and unloads many distinct
  models** would accumulate their weights in the pool until exit — effectively a leak for
  that pattern. There is currently **no eviction/bound**. Mitigation, if a reviewer
  pushes: add an opt-in byte-cap env var; eviction would be safe-by-construction (it only
  drops the pool's strong ref — any program still using a buffer keeps it alive via the
  shared_ptr). Not implemented because we have no evidence of the many-model-process
  workload. **Not tested for that pattern.**
- **Correctness rests on the byte-compare, not the hash.** This is deliberate: the FNV-1a
  hash only *finds candidates*; `same_bytes` is what makes a share correct. A reviewer
  should confirm the byte-compare is never bypassed (it isn't — every hit path goes
  through `same_bytes`).
- **VRAM pair.** This is the reclaim half. On its own it does nothing visible; it only
  matters when the EP co-resident program cache is on (which is what creates the 2×
  weight VRAM this removes). Review/ship the two together.
- **Layer:** the change is entirely in the GPU target (`src/targets/gpu/`), the correct
  place for device-memory logic — it does not touch MIGraphX core.
- **Thread-safety:** pool access is under a `std::mutex`; the Meyers singleton init is
  thread-safe. `gpu_literal::finalize` otherwise writes only its own instance.

---

## 6. Branches & artifacts

| Item | Value |
|---|---|
| Fork | `https://github.com/aditya-dl/AMDMIGraphX` |
| Branch | [`share-literals`](https://github.com/aditya-dl/AMDMIGraphX/tree/share-literals) |
| Mirror branch | `amd/dev/adilohia/share-literals` (same code, no README) |
| Commit | `732c09b96` (code) |
| Base | `develop` (`2b90a7914`) — cherry-picked clean, no conflicts |
| File | `src/targets/gpu/write_literals.cpp` (+88 / -1) |
| Flag | `MIGRAPHX_SHARE_LITERALS` — **default off** |
| PR status | none (not opened) |

---

## 7. How it relates to other tracks

- **Co-resident program cache (MIGraphX EP repo).** `ORT_MIGRAPHX_CORESIDENT_PROGRAMS`
  in `onnxruntime-ep-amdgpu` keeps both programs resident to avoid per-request reloads.
  That is what creates the 2× weight VRAM; **this change reclaims it.** They are a pair —
  see §4, §5. Separate repo, separate PR.
- **Parallel finalize (MIGraphX repo).** A separate change parallelizes program
  finalization (`MIGRAPHX_FINALIZE_PARALLEL`). Orthogonal to weight-sharing; different
  file, different concern.
- **hipGraph decode-capture (PR #5019).** A different effort on the *decode* compute path
  (graph capture/replay). Orthogonal; different files; no overlap with weight memory.
- **Build-coupling rule.** MIGraphX statically links rocMLIR. Build this against **stock
  rocMLIR**, not the experimental M=1-GEMV branch (it carries an int4-compile bug).
