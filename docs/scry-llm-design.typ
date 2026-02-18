// scry-llm: Technical Design Document
// GPT-2 Training Framework in Rust
// v0.2 — Post-Review Revision

#set document(
  title: "scry-llm: GPT-2 Training Framework in Rust",
  author: "esoc",
  date: auto,
)

#set page(
  paper: "a4",
  margin: (x: 2.4cm, y: 2.8cm),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: luma(120))
      scry-llm Technical Design Document — v0.2
      #h(1fr)
      #counter(page).display()
    ]
  },
)

#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(0.5em)
  text(size: 16pt, weight: "bold", it)
  v(0.3em)
}

#show heading.where(level: 2): it => {
  v(0.4em)
  text(size: 12pt, weight: "bold", it)
  v(0.2em)
}

#show raw.where(block: true): it => {
  set text(size: 9pt)
  block(fill: luma(245), inset: 10pt, radius: 3pt, width: 100%, it)
}

// ─────────────────────────── TITLE PAGE ───────────────────────────

#align(center)[
  #v(3cm)
  #text(size: 28pt, weight: "bold")[scry-llm]
  #v(0.3cm)
  #text(size: 14pt, fill: luma(80))[GPT-2 124M Training Framework in Rust]
  #v(0.5cm)
  #line(length: 40%, stroke: 0.5pt + luma(180))
  #v(0.5cm)
  #text(size: 11pt)[Technical Design Document]
  #v(0.2cm)
  #text(size: 10pt, fill: luma(100))[v0.2 — Post-Review Revision]
  #v(2cm)

  #table(
    columns: (auto, auto),
    stroke: none,
    align: (right, left),
    [*Author:*], [esoc],
    [*Target:*], [NVIDIA RTX 5070 Ti (16 GB, Blackwell)],
    [*Language:*], [Rust (MSRV 1.83.0)],
    [*Model:*], [GPT-2 124M (12L-12H-768D)],
    [*Parent:*], [scry workspace (`scry-engine`, `scry-chart`, `scry-learn`)],
  )
]

#pagebreak()

// ─────────────────────────── TOC ───────────────────────────

#outline(indent: 1.5em, depth: 3)

// ═══════════════════════════════════════════════════════════
= Executive Summary
// ═══════════════════════════════════════════════════════════

*scry-llm* is a new crate in the scry workspace that implements GPT-2 124M training from scratch in Rust, targeting the NVIDIA RTX 5070 Ti. The project has three concrete goals:

+ *A working autograd engine* — arena-allocated, tape-based reverse-mode automatic differentiation with a CUDA backend via `cudarc`. Custom fused kernels for transformer-specific operations; cuBLAS for GEMM.

+ *A trainable GPT-2 124M* — full BPE tokenizer, 12-layer transformer, mixed-precision training (BF16 compute, FP32 master weights), producing a model that generates coherent Rust code completions.

+ *Demonstrable performance* — target 40–55% model FLOPS utilization (MFU) on consumer hardware, with reproducible benchmarks and a clear comparison against PyTorch on equivalent hardware.

*Non-goals:* Distributed training, inference serving, RLHF, models larger than 350M parameters, AMD/Intel GPU support (v1).

*What this project is:* A systems engineering project that proves Rust's viability for ML training, builds deep CUDA expertise, and produces a clean reference implementation. The performance ceiling is the same hardware (cuBLAS) that PyTorch and llm.c use — this project will not be faster than either. The value is in demonstrating that a pure-Rust training pipeline with custom kernel fusion can reach competitive MFU without depending on any ML framework.

*What this project is not:* A production ML framework, a PyTorch replacement, or a novel training algorithm. The model architecture is standard GPT-2. The novelty is the implementation substrate.

// ═══════════════════════════════════════════════════════════
= Architecture Overview
// ═══════════════════════════════════════════════════════════

The system is structured as five layers, each building on the one below. This layering enables independent testing and benchmarking at each level.

== System Layers

```
┌─────────────────────────────────────────────────┐
│  Layer 5: Training Pipeline                      │
│  DataLoader → Forward → Loss → Backward → Step   │
├─────────────────────────────────────────────────┤
│  Layer 4: Model (GPT-2 124M)                     │
│  Embedding → 12× TransformerBlock → LMHead       │
├─────────────────────────────────────────────────┤
│  Layer 3: Autograd Engine                        │
│  Tape recording, backward pass, gradient accum   │
├─────────────────────────────────────────────────┤
│  Layer 2: Tensor Ops                             │
│  MatMul, Softmax, LayerNorm, GELU, CrossEntropy  │
├─────────────────────────────────────────────────┤
│  Layer 1: Device Abstraction (CUDA Backend)      │
│  cudarc + cuBLAS + custom PTX kernels            │
└─────────────────────────────────────────────────┘
```

== Crate Structure

The new crate `scry-llm` lives at `crates/scry-llm/` in the workspace:

```
crates/scry-llm/
├── Cargo.toml
├── src/
│   ├── lib.rs              # public API surface
│   ├── tensor/
│   │   ├── mod.rs          # Tensor<B: Backend> type
│   │   ├── shape.rs        # Shape, stride, broadcasting
│   │   ├── ops.rs          # operator trait definitions
│   │   └── view.rs         # zero-copy slicing, reshape
│   ├── autograd/
│   │   ├── mod.rs          # GradTape, TensorId
│   │   ├── tape.rs         # arena-based computation graph
│   │   ├── backward.rs     # reverse-mode traversal
│   │   └── ops.rs          # backward implementations per op
│   ├── backend/
│   │   ├── mod.rs          # Backend trait
│   │   ├── cpu.rs          # reference CPU backend
│   │   └── cuda/
│   │       ├── mod.rs      # CudaBackend struct
│   │       ├── device.rs   # cudarc device management
│   │       ├── memory.rs   # memory pool allocator
│   │       ├── blas.rs     # cuBLAS GEMM wrappers
│   │       └── kernels/    # custom .cu / PTX sources
│   │           ├── fused_attention.cu
│   │           ├── fused_layernorm.cu
│   │           ├── fused_gelu.cu
│   │           ├── fused_adamw.cu
│   │           ├── cross_entropy.cu
│   │           └── embed_lookup.cu
│   ├── nn/
│   │   ├── mod.rs          # Module trait
│   │   ├── linear.rs       # Linear layer
│   │   ├── embedding.rs    # token + position embeddings
│   │   ├── layernorm.rs    # LayerNorm
│   │   ├── attention.rs    # MultiHeadSelfAttention
│   │   ├── mlp.rs          # MLP (GELU sandwich)
│   │   ├── transformer.rs  # TransformerBlock = Attn + MLP
│   │   └── gpt2.rs         # full GPT2Model
│   ├── optim/
│   │   ├── mod.rs          # Optimizer trait
│   │   └── adamw.rs        # AdamW with fused CUDA kernel
│   ├── tokenizer/
│   │   ├── mod.rs          # Tokenizer trait
│   │   ├── bpe.rs          # byte-pair encoding trainer
│   │   └── vocab.rs        # merge table, encode/decode
│   ├── data/
│   │   ├── mod.rs          # Dataset, DataLoader
│   │   ├── mmap.rs         # memory-mapped token files
│   │   └── batch.rs        # batching + sequence packing
│   └── train/
│       ├── mod.rs          # TrainingConfig, Trainer
│       ├── schedule.rs     # LR schedulers (cosine, warmup)
│       ├── checkpoint.rs   # save/load model state
│       ├── mixed.rs        # BF16 precision management
│       └── metrics.rs      # loss tracking, MFU calculation
├── kernels/                # .cu source files (compiled at build time)
├── tests/
│   ├── grad_check.rs       # numerical gradient verification
│   ├── kernel_correctness.rs
│   └── training_smoke.rs
└── benches/
    ├── gemm_bench.rs       # GEMM throughput vs cuBLAS ceiling
    ├── attention_bench.rs  # fused vs unfused attention
    └── e2e_bench.rs        # full training step throughput
```

// ═══════════════════════════════════════════════════════════
= Layer 1: Device Abstraction & CUDA Backend
// ═══════════════════════════════════════════════════════════

== Backend Trait

All tensor operations are dispatched through a `Backend` trait, allowing CPU and CUDA implementations to coexist. The CPU backend serves as the reference for correctness testing.

The trait is split into focused sub-traits to avoid a monolithic interface where the CPU backend must stub out CUDA-specific fused ops:

```rust
/// Memory management — alloc, transfer, streams.
pub trait DeviceBackend: Send + Sync + 'static {
    type Storage: Clone;
    type Stream;

    fn alloc(&self, shape: &Shape, dtype: DType) -> Self::Storage;
    fn copy_to_host(&self, storage: &Self::Storage, dst: &mut [f32]);
    fn copy_from_host(&self, src: &[f32], storage: &mut Self::Storage);
    fn default_stream(&self) -> &Self::Stream;
    fn synchronize(&self, stream: &Self::Stream);
}

/// Core math ops — GEMM, elementwise, reductions.
pub trait MathBackend: DeviceBackend {
    fn gemm(
        &self, alpha: f32, a: &Self::Storage, b: &Self::Storage,
        beta: f32, c: &mut Self::Storage,
        m: usize, n: usize, k: usize,
        trans_a: bool, trans_b: bool,
    );
    fn gelu(&self, input: &Self::Storage, output: &mut Self::Storage);
    fn gelu_backward(&self, input: &Self::Storage, grad: &Self::Storage,
                     output: &mut Self::Storage);
    fn softmax(&self, input: &Self::Storage, output: &mut Self::Storage,
               axis: usize, shape: &Shape);
    fn layernorm(&self, input: &Self::Storage, gamma: &Self::Storage,
                 beta: &Self::Storage, output: &mut Self::Storage,
                 eps: f32, shape: &Shape);
    fn cross_entropy(&self, logits: &Self::Storage, targets: &[u32],
                     output: &mut Self::Storage, shape: &Shape);
}

/// Fused ops — optional, CUDA-only. CPU backend does not implement this.
/// Falls back to decomposed MathBackend calls when unavailable.
pub trait FusedBackend: MathBackend {
    fn fused_attention(
        &self, q: &Self::Storage, k: &Self::Storage, v: &Self::Storage,
        output: &mut Self::Storage,
        batch: usize, heads: usize, seq_len: usize, head_dim: usize,
        causal: bool,
    );
    fn fused_layernorm_residual(
        &self, input: &Self::Storage, residual: &Self::Storage,
        gamma: &Self::Storage, beta: &Self::Storage,
        output: &mut Self::Storage, eps: f32, shape: &Shape,
    );
    fn fused_adamw_step(
        &self, param: &mut Self::Storage, grad: &Self::Storage,
        m: &mut Self::Storage, v: &mut Self::Storage,
        lr: f32, beta1: f32, beta2: f32, eps: f32,
        weight_decay: f32, step: u32,
    );
}
```

All methods take `&self`, giving backends access to device handles, cuBLAS contexts, memory pools, and kernel caches. The `Stream` associated type enables explicit async overlap (see §3.3).

== CUDA Backend via cudarc

The CUDA backend wraps `cudarc` for device management and memory allocation. Key design decisions:

*cuBLAS for GEMM.* Matrix multiplication accounts for ~95% of training FLOPs. cuBLAS is hand-tuned by NVIDIA and achieves 70–80% of peak throughput. Re-implementing GEMM would cost months for a worse result.

*Custom kernels for everything else.* Operations like LayerNorm, GELU, softmax, attention, and AdamW are fused into custom CUDA kernels. This is where real performance gains live — not in faster matmul, but in fewer memory round-trips.

*Kernel compilation: NVRTC at runtime.* Kernels are written as `.cu` source strings bundled in the binary and compiled to PTX via NVRTC on first use, then cached. This avoids requiring `nvcc` in the build toolchain (only the CUDA driver + runtime are needed). Trade-off: ~1-2s startup cost on first run. Alternative (build-time `nvcc` via `build.rs`) adds build system complexity and requires the full CUDA toolkit at compile time — not worth it for 6 kernels.

*Bucketed caching allocator.* `cudaMalloc` / `cudaFree` are expensive (~1ms each). A simple free-list fragments under variable-size tensor allocs. Instead, we use a bucketed caching allocator: allocations are rounded up to the nearest size bucket (powers of 2 above 1 MB, fixed set of common sizes below). Freed blocks are cached per-bucket for reuse. Since GPT-2's computation graph is static, the allocation pattern stabilizes after the first training step — no fragmentation after warmup.

```rust
pub struct CudaBackend {
    device: Arc<cudarc::driver::CudaDevice>,
    blas: cudarc::cublas::CudaBlas,
    alloc: CachingAllocator,
    kernels: KernelCache,       // NVRTC-compiled PTX, cached per-signature
    compute_stream: CudaStream, // primary compute stream
    transfer_stream: CudaStream, // H2D/D2H overlap
}

struct CachingAllocator {
    /// Cached free blocks, bucketed by size class.
    /// Key = size bucket, Value = Vec of (device_ptr, actual_size).
    cache: HashMap<usize, Vec<(CudaSlice<u8>, usize)>>,
    /// Total bytes currently allocated on device.
    allocated: usize,
    /// High-water mark for memory tracking.
    peak: usize,
}

impl CachingAllocator {
    fn size_bucket(size: usize) -> usize {
        if size <= 1 << 20 {
            size.next_power_of_two()  // exact power-of-2 below 1 MB
        } else {
            (size + (1 << 20) - 1) & !((1 << 20) - 1)  // 1 MB alignment above
        }
    }
}
```

== CUDA Stream Management

Two streams enable overlapping data transfer with computation:
- `compute_stream`: all kernel launches and cuBLAS calls
- `transfer_stream`: H2D uploads (next batch prefetch) and D2H downloads (loss reporting)

Events synchronize between streams when a transfer must complete before compute begins (batch ready) or vice versa (checkpoint save). This is required for the M4 milestone (pipeline overlap) but the two-stream design is established from Phase 3 to avoid retrofit.

== Custom Kernel Strategy

Seven fused kernels cover the operations where cuBLAS doesn't help:

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [*Kernel*], [*What it fuses*], [*Mem savings*],
  [`fused_attention_fwd`], [Q·K^T → scale → mask → softmax → ·V \ (FlashAttention-2 tiling, forward pass)], [O(N) vs O(N²)],
  [`fused_attention_bwd`], [Attention backward: dQ, dK, dV without materializing full N×N attention matrix], [O(N) vs O(N²)],
  [`fused_layernorm`], [mean → var → normalize → scale → shift + residual add], [2× fewer reads],
  [`fused_gelu`], [GELU activation with backward pass in one kernel], [2× fewer reads],
  [`fused_adamw`], [param update: m,v,weight_decay,lr all in one pass], [4× fewer reads],
  [`cross_entropy`], [logits → log_softmax → nll_loss in one pass], [no materialized probs],
  [`embed_lookup`], [batched gather from embedding table], [coalesced access],
)

Note: `fused_attention_fwd` and `fused_attention_bwd` are separate kernels because the backward pass requires different tiling and additional intermediate storage (row-max and row-sum from the forward softmax). Both are part of the FlashAttention-2 algorithm.

Each kernel is written in CUDA C, compiled to PTX via NVRTC at runtime (first use), then cached. Kernel source is embedded in the binary as string constants — no external `.cu` files needed at runtime.

// ═══════════════════════════════════════════════════════════
= Layer 2: Tensor System
// ═══════════════════════════════════════════════════════════

== Tensor Representation

```rust
pub struct Tensor<B: Backend> {
    id: TensorId,               // unique ID for autograd graph
    storage: B::Storage,        // backend-specific data
    shape: Shape,               // dimensions
    dtype: DType,               // F32 or BF16
    requires_grad: bool,
}

pub struct Shape {
    dims: ArrayVec<usize, 4>,   // max 4D: [batch, heads/seq, seq, dim]
    strides: ArrayVec<usize, 4>,
}

#[derive(Clone, Copy)]
pub enum DType { F32, BF16 }
```

*Why max 4D?* GPT-2 attention tensors are 4D `[batch, heads, seq, head_dim]`. No transformer operation requires more. Using `ArrayVec<usize, 4>` keeps shapes on the stack — no heap allocation for the most common operation in the entire framework.

== Broadcasting Rules

Standard NumPy-style broadcasting: dimensions are compared right-to-left, each must be equal or one of them is 1. Broadcasting is implemented via stride tricks (set stride to 0 for broadcast dimensions), avoiding data copies.

== Mixed Precision (BF16)

Training uses BF16 for forward and backward passes (2× less memory, 2× faster on tensor cores), with FP32 master weights maintained by the optimizer.

*Why BF16 over FP16:* Blackwell tensor cores support both at identical throughput. BF16 has the same exponent range as FP32 (8 exponent bits vs FP16's 5), which means gradients almost never underflow. This eliminates the need for dynamic loss scaling — a significant simplification that removes an entire class of numerical debugging. The trade-off is slightly less mantissa precision (7 bits vs FP16's 10), which is irrelevant for training where gradient noise dominates.

```
Forward:   BF16 activations, BF16 weights (cast from FP32)
Backward:  BF16 gradients
Optimizer: FP32 master weights, FP32 momentum/variance
```

Loss scaling is not required with BF16 but a simple static scaler is kept as a safety net (scale=1.0 by default, only activated if NaN detected). This is far simpler than FP16's mandatory dynamic loss scaling.

// ═══════════════════════════════════════════════════════════
= Layer 3: Autograd Engine
// ═══════════════════════════════════════════════════════════

== Design: Arena-Based Tape

The autograd engine uses a *tape* (ordered list of operations) rather than a DAG. This is the most Rust-friendly approach — no `Rc<RefCell<>>`, no reference cycles, no garbage collection.

```rust
pub struct GradTape<B: Backend> {
    nodes: Vec<TapeNode<B>>,     // arena: index = TensorId
    grads: Vec<Option<B::Storage>>,  // accumulated gradients
}

struct TapeNode<B: Backend> {
    op: Operation,
    inputs: ArrayVec<TensorId, 3>,  // most ops have 1-2 inputs
    shape: Shape,
    saved: SavedData<B>,             // tensors needed for backward
}

/// What each op saves for its backward pass. Explicitly enumerated
/// because this drives peak memory — every field here is a GPU allocation
/// that lives until backward reaches this node.
enum SavedData<B: DeviceBackend> {
    /// MatMul: save both inputs (needed for dA = dC @ B^T, dB = A^T @ dC)
    MatMul { a: B::Storage, b: B::Storage, trans_a: bool, trans_b: bool },
    /// Add: nothing saved (backward is identity)
    Add,
    /// LayerNorm: save normalized output, gamma, mean, rstd
    LayerNorm { normalized: B::Storage, gamma: B::Storage,
                mean: B::Storage, rstd: B::Storage, eps: f32 },
    /// GELU: save input (needed for GELU'(x))
    Gelu { input: B::Storage },
    /// Softmax: save output (backward: dX = Y * (dY - sum(dY * Y)))
    Softmax { output: B::Storage, axis: usize },
    /// CrossEntropy: save log-probs and targets
    CrossEntropy { log_probs: B::Storage, targets: Vec<u32> },
    /// Embedding: save indices (for scatter-add gradient)
    Embedding { indices: Vec<u32> },
    /// FusedAttention: save Q, K, V, output, and row-wise logsumexp
    /// (FlashAttention-2 backward needs these to avoid rematerializing attention)
    FusedAttention { q: B::Storage, k: B::Storage, v: B::Storage,
                     output: B::Storage, lse: B::Storage, causal: bool },
    /// Checkpoint boundary: only saves input tensor. Intermediates are
    /// recomputed during backward (see §3.4).
    Checkpoint { input: B::Storage, block_start: usize, block_end: usize },
}
```

== Backward Pass

Reverse-mode autodiff traverses the tape from loss back to parameters:

```rust
impl<B: Backend> GradTape<B> {
    pub fn backward(&mut self, loss_id: TensorId) {
        // Seed: dL/dL = 1.0
        self.grads[loss_id.0] = Some(B::ones(&self.nodes[loss_id.0].shape));

        // Walk tape in reverse
        for i in (0..=loss_id.0).rev() {
            let grad_output = match self.grads[i].take() {
                Some(g) => g,
                None => continue,  // not on the path from loss
            };

            let node = &self.nodes[i];
            let input_grads = node.op.backward::<B>(
                &node.saved,
                &grad_output,
            );

            // Accumulate into input gradients
            for (input_id, grad) in node.inputs.iter().zip(input_grads) {
                accumulate::<B>(&mut self.grads[input_id.0], grad);
            }
        }
    }
}
```

== Why tape over DAG?

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [], [*Tape (chosen)*], [*DAG*],
  [Ownership], [Vec owns all nodes. No Rc/Arc.], [Shared ownership required.],
  [Traversal], [Simple reverse iteration.], [Topological sort needed.],
  [Memory], [Contiguous, cache-friendly.], [Pointer-chasing.],
  [Rust fit], [Natural. No interior mutability.], [Needs RefCell or unsafe.],
  [Limitation], [No control flow (if/loops in graph).], [Supports dynamic graphs.],
)

GPT-2 has a completely static computation graph — every forward pass executes the same operations in the same order. The tape's limitation (no dynamic control flow) costs us nothing here.

== Gradient Checkpointing

For memory efficiency, we don't save all intermediate activations. Instead, each transformer block is a *checkpoint boundary*: during backward, we re-run the forward pass for that block to recompute activations. This trades ~30% more compute for ~60% less activation memory.

*Interaction with the tape:* Checkpointing complicates the "simple reverse iteration" story. During backward, when the traversal hits a `Checkpoint` node, it must:

+ Re-run the forward pass for that block, recording ops into a *temporary recompute tape*
+ Run backward on the recompute tape to produce gradients for the block's inputs
+ Discard the recompute tape and continue the main backward traversal

This means the tape is not a single flat `Vec` during backward. Implementation uses a *segmented tape*: the main tape holds `Checkpoint` sentinel nodes where blocks were, and each block's recomputation produces an ephemeral sub-tape that is consumed immediately.

```rust
impl<B: MathBackend> GradTape<B> {
    fn backward_through_checkpoint(
        &self, node: &CheckpointNode<B>, grad_output: &B::Storage,
        backend: &B,
    ) -> B::Storage {
        // 1. Recompute: re-run forward for this block into a temp tape
        let mut recompute_tape = GradTape::new();
        let block_output = self.recompute_block(
            node.block_start, node.block_end,
            &node.input, &mut recompute_tape, backend,
        );

        // 2. Backward on the recompute tape
        recompute_tape.seed_grad(block_output.id, grad_output.clone());
        recompute_tape.backward(block_output.id);

        // 3. Extract gradient w.r.t. block input, discard recompute tape
        let input_grad = recompute_tape.take_grad(node.input_id);
        // recompute_tape is dropped here — memory freed
        input_grad
    }
}
```

This is the most complex part of the autograd engine and must be tested extensively: verify that checkpointed backward produces identical gradients to non-checkpointed backward on the same model.

// ═══════════════════════════════════════════════════════════
= Layer 4: GPT-2 Model
// ═══════════════════════════════════════════════════════════

== Architecture Specification

GPT-2 124M ("small") parameters:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [*Hyperparameter*], [*Value*], [*Notes*],
  [Layers ($n_"layer"$)], [12], [],
  [Heads ($n_"head"$)], [12], [],
  [Embedding dim ($d_"model"$)], [768], [],
  [Head dim ($d_"head"$)], [64], [$d_"model" / n_"head"$],
  [FFN inner dim ($d_"ff"$)], [3072], [$4 times d_"model"$],
  [Vocab size ($|V|$)], [50257], [BPE],
  [Context length ($T$)], [1024], [positional embedding limit],
  [Parameters], [~124M], [~500 MB FP32],
)

== Module Trait

All layers implement a common interface. The tape is `Option` to support both training (tape recording) and inference (no tape, no gradient storage):

```rust
pub trait Module<B: MathBackend> {
    fn forward(
        &self, input: &Tensor<B>,
        tape: Option<&mut GradTape<B>>,  // None = inference mode
    ) -> Tensor<B>;
    fn parameters(&self) -> Vec<&Tensor<B>>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<B>>;
}
```

When `tape` is `None`, ops execute without recording and `SavedData` is not allocated. This cleanly separates training from inference without needing a separate code path or type-level flag.

== Model Components

*Embedding layer.* Token embedding ($|V| times d_"model"$) + learned positional embedding ($T times d_"model"$). Summed, not concatenated.

*Transformer block (×12).* Pre-norm architecture (LayerNorm before attention and MLP, matching GPT-2):

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

*Multi-head self-attention.* Single fused QKV projection ($d_"model" arrow.r 3 dot d_"model"$), split into 12 heads, FlashAttention-2 kernel for the attention computation, output projection ($d_"model" arrow.r d_"model"$).

*MLP.* Two linear layers with GELU activation: $d_"model" arrow.r d_"ff" arrow.r d_"model"$. The GELU and first linear can be fused.

*LM Head.* Final LayerNorm → linear projection to vocab size. Weight-tied with token embedding matrix (standard practice, saves 38M parameters).

== Weight Initialization

Following GPT-2:
- All weights: $cal(N)(0, 0.02)$
- Residual projections (attention out, MLP out): scaled by $1 / sqrt(2 dot n_"layer")$
- Biases: zero
- LayerNorm: $gamma = 1$, $beta = 0$

// ═══════════════════════════════════════════════════════════
= Layer 5: Training Pipeline
// ═══════════════════════════════════════════════════════════

== Data Pipeline

=== BPE Tokenizer

Trained from scratch on the target corpus (Rust source code). Algorithm:

+ Initialize vocabulary with 256 byte tokens
+ Count all adjacent token pairs in corpus
+ Merge the most frequent pair into a new token
+ Repeat until vocabulary reaches target size (32,768 for code — smaller than GPT-2's 50,257 since code has less lexical variety)
+ Store merge table for encoding, reverse table for decoding

The tokenizer is a separate binary/step — run once, produces a vocab file and a tokenized corpus.

=== Pre-tokenized Dataset

Training data is pre-tokenized and stored as a flat binary file of `u16` token IDs (vocab < 65k). Memory-mapped at training time for zero-copy access.

```rust
pub struct MmapDataset {
    mmap: memmap2::Mmap,       // memory-mapped token file
    seq_len: usize,            // 1024 for GPT-2
    n_sequences: usize,
}

impl MmapDataset {
    fn get_batch(&self, indices: &[usize]) -> (Vec<u16>, Vec<u16>) {
        // returns (inputs[:-1], targets[1:]) for each sequence
    }
}
```

=== DataLoader

Shuffled batching with:
- Random sequence offset within documents (not just document boundaries)
- Sequence packing: short documents concatenated with `<|endoftext|>` separator
- Prefetching: next batch loaded to GPU while current batch trains

== Optimizer: AdamW

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)
θ_t = θ_{t-1} - lr · (m̂_t / (√v̂_t + ε) + λ · θ_{t-1})
```

Hyperparameters (GPT-2 standard):

#table(
  columns: (auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [$beta_1$], [0.9],
  [$beta_2$], [0.95],
  [$epsilon$], [1e-8],
  [$lambda$ (weight decay)], [0.1],
  [Peak learning rate], [6e-4],
  [Warmup steps], [2000],
  [Schedule], [Cosine decay to 6e-5],
  [Batch size], [64 sequences × 1024 tokens = 65,536 tokens/step],
  [Gradient accumulation], [as needed to reach batch size],
)

The entire AdamW update (bias correction, momentum, variance, weight decay, param update) is fused into a single CUDA kernel — one read of each parameter and state tensor, one write. 4× fewer global memory accesses than the naive 4-kernel approach.

== Training Loop (Pseudocode)

```rust
let mut model = Gpt2Model::new(&config, &backend);
let mut optim = AdamW::new(model.parameters_mut(), &optim_config);
let loader = DataLoader::new(&dataset, batch_size, seq_len);

for step in 0..config.max_steps {
    let lr = cosine_schedule(step, config.warmup, config.max_steps);
    optim.set_lr(lr);

    let mut accum_loss = 0.0;

    // Gradient accumulation: each micro-batch gets its own tape.
    // Gradients are accumulated in FP32 in the optimizer state to
    // avoid BF16 summation overflow across micro-batches.
    for _micro in 0..config.grad_accum_steps {
        let mut tape = GradTape::new();
        let (input, target) = loader.next_batch();
        let logits = model.forward(&input, Some(&mut tape));
        let loss = cross_entropy(&logits, &target, &mut tape);
        tape.backward(loss.id);

        // Accumulate gradients (FP32) into optimizer buffers
        optim.accumulate_grads(&tape, 1.0 / config.grad_accum_steps as f32);
        accum_loss += loss.item();
        // tape is dropped here — activation memory freed per micro-batch
    }

    // NaN check (safety net — BF16 rarely needs this)
    if !optim.grads_contain_nan() {
        optim.step();
    } else {
        log_warning(step, "NaN in gradients, skipping step");
    }

    optim.zero_grad();

    if step % config.log_every == 0 {
        log_metrics(step, accum_loss, lr);
    }
    if step % config.save_every == 0 {
        save_checkpoint(&model, &optim, step);
    }
}
```

Key difference from v0.1: each micro-batch uses a fresh tape (freed after backward), and gradient accumulation happens in FP32 inside the optimizer. This avoids both the BF16 accumulation overflow issue and the memory cost of keeping all micro-batch tapes alive simultaneously.

== Learning Rate Schedule

Cosine decay with linear warmup:

$ "lr"(t) = cases(
  "lr"_"peak" dot t / t_"warmup" & "if" t < t_"warmup",
  "lr"_"min" + 1/2 ("lr"_"peak" - "lr"_"min")(1 + cos(pi dot (t - t_"warmup") / (t_"max" - t_"warmup"))) & "otherwise"
) $

// ═══════════════════════════════════════════════════════════
= Tokenizer Design
// ═══════════════════════════════════════════════════════════

== Why Custom BPE

GPT-2's original tokenizer was trained on WebText (general English). Rust code has very different token distributions — `fn`, `mut`, `impl`, `::`, `->`, `pub`, `&str` should all be single tokens. A code-specific BPE with 32K vocab achieves better compression ratios (fewer tokens per line of code = more context in the 1024-token window).

== Training Algorithm

```
function train_bpe(corpus: bytes, vocab_size: u32) -> MergeTable:
    vocab = {0..255}  // initial byte-level tokens
    tokens = [byte for byte in corpus]

    while |vocab| < vocab_size:
        // Count all adjacent pairs
        pair_counts = count_pairs(tokens)
        best_pair = argmax(pair_counts)

        // Merge all occurrences
        new_token = |vocab|
        vocab.insert(new_token, best_pair)
        tokens = merge_all(tokens, best_pair, new_token)

    return vocab, merge_table
```

*Optimization:* The naive pair-counting is O(n) per merge, with ~32K merges = O(32K·n). For a multi-GB corpus this is slow. We use an indexed pair table that updates incrementally on each merge — only re-count pairs adjacent to merged positions. Reduces wall-clock from hours to minutes.

== Encoding / Decoding

```rust
pub struct BpeTokenizer {
    merges: Vec<(u32, u32)>,        // ordered merge rules
    vocab: Vec<Vec<u8>>,            // token_id → bytes
    token_to_id: HashMap<Vec<u8>, u32>,
}

impl BpeTokenizer {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        for &(a, b) in &self.merges {
            tokens = merge_pass(&tokens, a, b);
        }
        tokens
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter()
            .flat_map(|&t| self.vocab[t as usize].iter().copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
```

// ═══════════════════════════════════════════════════════════
= Performance Targets & Analysis
// ═══════════════════════════════════════════════════════════

== MFU Targets

Model FLOPS Utilization (MFU) measures what fraction of the GPU's theoretical peak is actually used for model computation.

$ "MFU" = (6 dot N dot B dot T) / ("GPU peak FLOPS" dot "wall time per step") $

Where $N$ = parameters, $B$ = batch size, $T$ = sequence length.

*GPU peak FLOPS baseline:* The RTX 5070 Ti's BF16 dense tensor core throughput is ~88 TFLOPS. The ~176 TFLOPS figure cited in some sources includes structured sparsity (2:4), which cannot be used for dense training workloads. All MFU calculations below use 88 TFLOPS as the denominator.

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [*Milestone*], [*MFU*], [*Effective TFLOPS*], [*What it means*],
  [M1: Correct training], [10–20%], [~9–18], [Naive kernels, no fusion],
  [M2: cuBLAS GEMM], [30–40%], [~26–35], [cuBLAS for matmul, naive rest],
  [M3: Fused kernels], [40–55%], [~35–48], [FlashAttn + fused ops],
  [M4: Optimized pipeline], [50–60%], [~44–53], [Overlap compute/transfer],
  [Practical ceiling], [60–70%], [~53–62], [PyTorch w/ torch.compile],
)

*Realistic target: M3 (40–55% MFU).* 50% would be a strong result. 55% would be excellent. For reference, llm.c achieves ~40% on A100 for GPT-2 124M, and nanoGPT reports 29–64% depending on configuration and how peak FLOPS is counted.

== Memory Budget (16 GB)

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [*Component*], [*FP32*], [*Mixed Precision (BF16)*],
  [Model weights], [~500 MB], [~250 MB (BF16) + 500 MB (FP32 master)],
  [Gradients], [~500 MB], [~250 MB (BF16)],
  [Optimizer state (m, v)], [~1000 MB], [~1000 MB (FP32)],
  [Activations (batch=8, no ckpt)], [~4000 MB], [~2000 MB],
  [Activations (batch=8, ckpt)], [~1500 MB], [~800 MB],
  [cuBLAS workspace], [~128 MB], [~128 MB],
  [CUDA context + libraries], [~300 MB], [~300 MB],
  [Allocator overhead + fragmentation], [~200 MB], [~200 MB],
  [*Total*], [*~8.6 GB*], [*~3.9 GB*],
)

Mixed precision + gradient checkpointing leaves ~12.1 GB free on 16 GB. Realistic usable memory after CUDA context and library loading is ~10–11 GB. This allows batch size 8 comfortably. With gradient accumulation over 8 micro-batches, effective batch = 64 sequences = 65K tokens/step.

== Training Time Estimate

At 2B tokens, batch size 64, sequence length 1024, using 88 TFLOPS dense BF16 as peak:

#table(
  columns: (auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [Total tokens], [2,000,000,000],
  [Tokens per step], [65,536],
  [Total steps], [~30,500],
  [FLOPs per step], [$6 times 124 times 10^6 times 65536 approx 4.87 times 10^13$],
  [At M3 (45% MFU = ~40 eff. TFLOPS)], [~1.22s per step],
  [*Total training time (45% MFU)*], [*~10.3 hours*],
  [At M3 (55% MFU = ~48 eff. TFLOPS)], [~1.01s per step],
  [*Total training time (55% MFU)*], [*~8.6 hours*],
)

For context: llm.c trains GPT-2 124M on a single A100 (312 TFLOPS peak) in ~90 minutes at higher absolute throughput. The 5070 Ti has ~3.5× less raw TFLOPS than an A100, making 8–10 hours the realistic range for a well-optimized consumer-GPU training run.

// ═══════════════════════════════════════════════════════════
= Verification Strategy
// ═══════════════════════════════════════════════════════════

== Correctness Tiers

*Tier 1: Unit-level gradient checking.* For every autograd operation, compare analytical gradients against numerical finite-difference gradients ($epsilon = 10^(-5)$). Must match to relative tolerance $< 10^(-4)$ for FP32.

```rust
fn grad_check<B: Backend>(f: impl Fn(&Tensor<B>) -> Tensor<B>,
                           x: &Tensor<B>, eps: f64) -> f64 {
    let analytical = {
        let mut tape = GradTape::new();
        let y = f(x, &mut tape);
        tape.backward(y.id);
        tape.grad(x.id)
    };

    let numerical = {
        // (f(x + eps) - f(x - eps)) / (2 * eps)  per element
    };

    max_relative_error(&analytical, &numerical)
}
```

*Tier 2: Kernel correctness.* Every CUDA kernel is tested against the CPU backend on identical inputs. Random inputs, seeded for reproducibility. Comparison uses ULP-aware relative error (expect ~1e-6 for FP32, ~1e-3 for BF16) — not raw epsilon, because GPU parallel reductions have different floating-point associativity than CPU serial reductions.

*Tier 3: Known-answer tests.* Load OpenAI's GPT-2 pretrained weights (requires a weight loader that reads TF/PyTorch checkpoint format and maps parameter names), run inference on fixed prompts, verify logits match to BF16 tolerance. This validates the entire model architecture is correctly assembled. Budget 1–2 days for the weight loading plumbing.

*Tier 4: Training convergence.* Train on a tiny synthetic dataset (1000 sequences of repeated patterns). Loss must reach < 0.01 within 500 steps. If it doesn't, something is wrong with gradients or the optimizer.

*Tier 5: Reproduction benchmark.* Train on OpenWebText for a fixed number of steps, compare loss curve against Karpathy's nanoGPT reference implementation at the same step count. Loss should be within 5%.

== Benchmark Protocol

Following scry-learn conventions:
- `std::hint::black_box()` for all timing
- 2+ warmup iterations before measurement
- Deterministic RNG (`fastrand::Rng::with_seed(42)`)
- Report: throughput (tokens/sec), MFU (%), wall time, memory peak

// ═══════════════════════════════════════════════════════════
= Risk Analysis
// ═══════════════════════════════════════════════════════════

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [*Risk*], [*Impact*], [*Mitigation*],
  [Autograd bugs], [Silent wrong gradients → model doesn't converge], [Tier 1 grad checks on every op before integration],
  [CUDA kernel bugs], [Wrong results or GPU hangs], [CPU reference comparison, small-input manual verification],
  [Numerical instability], [NaN/Inf during training], [BF16 (wide dynamic range), careful LayerNorm epsilon],
  [cudarc API churn (*HIGH*)], [cudarc 0.16.x has rewritten API from 0.12; maintainer transferred], [Target 0.16.x explicitly, vendor crate if unstable],
  [Blackwell kernel crashes (*HIGH*)], [FlashAttention upstream has open SM_100/SM_120 crash issues], [Require CUDA Toolkit 12.8+, target `compute_100`, test on hardware in Phase 3 not Phase 4],
  [SM_90 fallback is not viable], [Hopper instruction scheduling on Blackwell = degraded perf], [Do not fall back — write SM_100 kernels or use unfused path],
  [Build times], [cudarc + NVRTC + full workspace = 2-5 min clean builds], [Separate `scry-llm-kernels` crate, incremental builds],
  [CPU/CUDA numerical divergence], [Parallel GPU reduction ≠ serial CPU reduction at float level], [ULP-aware comparison, not raw epsilon; expect ~1e-6 relative error],
  [Scope creep], [Adding features before core works], [Hard milestone gates: no Layer N+1 until Layer N passes all tests],
  [Training data quality], [Garbage in, garbage out], [Curate dataset carefully, validate tokenizer compression ratio],
  [FlashAttention complexity], [Single hardest kernel; 1-3 weeks alone], [Start with unfused attention (cuBLAS), add FA-2 as optimization in Phase 4],
)

== Profiling Strategy

MFU gaps cannot be debugged without profiling. Required tooling:

- *CUDA events* for per-kernel timing (inserted around each kernel launch, negligible overhead)
- *nsys* (NVIDIA Nsight Systems) for timeline profiling — identifies gaps between kernel launches, synchronization stalls, memory transfer bottlenecks
- *ncu* (NVIDIA Nsight Compute) for kernel-level analysis — occupancy, memory throughput, compute throughput per kernel
- *Memory high-water tracking* via the caching allocator's `peak` field — reported each step during initial profiling runs

Profiling is not optional. It is part of Phase 3 (first CUDA runs) and Phase 4 (kernel optimization).

// ═══════════════════════════════════════════════════════════
= Implementation Roadmap
// ═══════════════════════════════════════════════════════════

Strict dependency order. Each phase has a gate — must pass before next begins.

== Phase 1: Tensor + CPU Backend (Foundation)
*Gate: all operations pass gradient checking on CPU.*

- `Tensor<CpuBackend>` with shape, strides, broadcasting
- All ops: matmul, add, softmax, layernorm, gelu, cross_entropy, embedding lookup
- Autograd tape with backward pass
- Comprehensive gradient check test suite

== Phase 2: GPT-2 on CPU (Correctness)
*Gate: loss decreases on synthetic data; logits match OpenAI checkpoint within tolerance.*

- All `nn` modules: Linear, Embedding, LayerNorm, Attention, MLP, TransformerBlock, Gpt2Model
- AdamW optimizer (CPU)
- Load pretrained GPT-2 weights, verify inference
- Tiny-data training convergence test

== Phase 3: CUDA Backend (Performance)
*Gate: CUDA outputs match CPU outputs to ULP-aware tolerance (~1e-6 relative) on all ops. Profiling baseline established.*

- cudarc 0.16.x device management + caching allocator
- cuBLAS GEMM wrapper (BF16 and FP32)
- Two-stream setup (compute + transfer)
- Port all ops to CUDA (naive kernels first)
- Backend-switching test: same training run on CPU and CUDA produces equivalent loss curves (not identical — floating point associativity differs)
- nsys profiling of first training steps to establish baseline MFU
- Test on actual 5070 Ti hardware — do not defer hardware testing

== Phase 4: Custom Kernels (Optimization)
*Gate: fused kernels match unfused outputs to BF16 tolerance. MFU ≥ 40%.*

- FlashAttention-2 forward + backward kernels (SM_100 target)
- Fused LayerNorm + residual
- Fused GELU
- Fused AdamW
- Fused cross-entropy
- Benchmark suite: per-kernel (via ncu) and end-to-end (via nsys)
- If FlashAttention proves too complex for v1, fall back to unfused attention with cuBLAS and accept lower MFU

== Phase 5: Training Pipeline (Production)
*Gate: successful training run on real data, generating coherent Rust code.*

- Tokenizer integration (use `tiktoken-rs` or `tokenizers` crate for v1; custom BPE is a stretch goal)
- Memory-mapped data loading
- BF16 mixed precision (simplified — no dynamic loss scaler)
- Gradient checkpointing with segmented tape
- Cosine LR schedule + warmup
- Checkpointing (save/resume)
- Training metrics logging (loss, MFU, memory peak, tokens/sec)

== Phase 6: Evaluation & Polish
*Gate: documented, benchmarked, reproducible.*

- Sample generation (temperature, top-k, top-p)
- Perplexity evaluation on held-out test set
- MFU benchmark against nanoGPT/PyTorch
- Documentation

// ═══════════════════════════════════════════════════════════
= Timeline Estimate
// ═══════════════════════════════════════════════════════════

CUDA kernel development is qualitatively different from CPU Rust — `printf` debugging, no stack traces, race conditions that manifest only at certain block sizes. Phase 3 and 4 will take longer than expected.

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [*Phase*], [*Estimate*], [*Notes*],
  [Phase 1: Tensor + CPU autograd], [1–2 weeks], [Shape/stride logic is fiddly. Grad checking will find bugs.],
  [Phase 2: GPT-2 on CPU], [1–2 weeks], [Layer assembly is straightforward. Weight loading is tedious format wrangling.],
  [Phase 3: CUDA backend], [2–4 weeks], [cudarc API, memory pool, host/device sync. This is where it gets hard.],
  [Phase 4: Custom kernels], [4–8 weeks], [FlashAttention-2 alone is 1–3 weeks. Fused kernel debugging is brutal.],
  [Phase 5: Training pipeline], [1–2 weeks], [Mostly plumbing.],
  [Phase 6: Evaluation], [1 week], [],
  [*Total*], [*10–19 weeks*], [*3–5 months. Phase 3+4 are where the schedule slips.*],
)

// ═══════════════════════════════════════════════════════════
= Data Sourcing
// ═══════════════════════════════════════════════════════════

Training on "Rust source code" requires a concrete data plan:

*Sources:*
- The Stack v2 (Rust subset): ~5–10 GB of deduplicated Rust source, permissively licensed
- crates.io source archives: ~15–20 GB total, need deduplication
- The scry workspace itself: ~few MB (too small alone, but included for domain adaptation)

*Preparation:*
- Deduplicate at file level (exact hash) and near-duplicate level (MinHash)
- Filter: remove auto-generated code (build scripts, bindings), vendored dependencies, files >100KB
- Target: 2B tokens post-tokenization, requiring ~8–10 GB of cleaned source

*Licensing:* Only include code under permissive licenses (MIT, Apache-2.0, BSD). The Stack v2 provides license metadata per file.

*Validation:* Measure tokenizer compression ratio on held-out Rust code. Target: fewer than 2 tokens per whitespace-delimited word (natural language BPE is ~1.3; code is denser).

// ═══════════════════════════════════════════════════════════
= Appendix A: Dependency Table
// ═══════════════════════════════════════════════════════════

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [*Crate*], [*Version*], [*Purpose*],
  [`cudarc`], [0.16.x], [CUDA driver API, cuBLAS bindings, NVRTC],
  [`memmap2`], [0.9+], [Memory-mapped dataset files],
  [`bytemuck`], [1.x], [Safe transmute for GPU buffer uploads],
  [`half`], [2.x], [BF16 type (`bf16`) for host-side conversions],
  [`fastrand`], [2.x], [Deterministic RNG for data shuffling],
  [`serde`], [1.x], [Checkpoint serialization (optional)],
  [`arrayvec`], [0.7], [Stack-allocated small vectors for shapes],
  [`thiserror`], [2.x], [Error types],
  [`tiktoken-rs`], [0.6+], [BPE tokenizer for v1 (custom BPE is stretch goal)],
)

No dependency on any ML framework. No PyTorch, no ONNX, no tch-rs, no candle, no burn.

// ═══════════════════════════════════════════════════════════
= Appendix B: GPT-2 124M Parameter Breakdown
// ═══════════════════════════════════════════════════════════

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt + luma(200),
  inset: 6pt,
  [*Component*], [*Shape*], [*Count*], [*%*],
  [Token embedding], [$50257 times 768$], [38,597,376], [31.1%],
  [Position embedding], [$1024 times 768$], [786,432], [0.6%],
  [Per-block QKV proj], [$768 times 2304 + 2304$], [$times$ 12 = 21,268,224], [17.2%],
  [Per-block attn out], [$768 times 768 + 768$], [$times$ 12 = 7,087,872], [5.7%],
  [Per-block MLP fc1], [$768 times 3072 + 3072$], [$times$ 12 = 28,348,416], [22.9%],
  [Per-block MLP fc2], [$3072 times 768 + 768$], [$times$ 12 = 28,320,768], [22.9%],
  [Per-block LayerNorms], [$768 times 2 times 2$], [$times$ 12 = 36,864], [< 0.1%],
  [Final LayerNorm], [$768 times 2$], [1,536], [< 0.1%],
  [LM head (tied)], [—], [0 (shares token emb)], [—],
  [*Total*], [], [*124,447,488*], [*100%*],
)

// ═══════════════════════════════════════════════════════════
= Appendix C: Reference Implementations
// ═══════════════════════════════════════════════════════════

For cross-validation during development:

- *nanoGPT* (Karpathy) — minimal PyTorch GPT-2 training. Loss curve reference.
- *llm.c* (Karpathy) — GPT-2 training in C/CUDA. Kernel reference.
- *cudarc examples* — CUDA Rust API usage patterns.
- *FlashAttention-2 paper* (Dao, 2023) — fused attention algorithm.
- *GPT-2 paper* (Radford et al., 2019) — architecture specification.

These serve as *correctness oracles*, not as code to port. The implementation is original.
