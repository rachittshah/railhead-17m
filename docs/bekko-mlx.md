# Bekko System One v0 17M in MLX

`rh17m.bekko_mlx` is a native MLX port of
[`hotchpotch/bekko-system-one-v0-17m`](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m) at revision
`b886a1f9b91f4e8368d7830080d52d955e2c8dfa` (the revision pinned by `decision_mlx.bekko_provider`). It covers
inference and differentiable batched training. Inputs are token-identical to the published runtime
(`inference_v0.py`), and outputs match it to within 2e-6 in probability.

```python
from rh17m.bekko_mlx import load, save, choice_probs, encode_choice, collate, make_optimizer, make_train_step

model = load()                                   # fp32; load(dtype=mx.bfloat16) for bf16 weights
probs = choice_probs(model, None, state_json, "Pick the best action.", [("route_1", "..."), ("hold", "...")])

tok = model.tokenizer                            # BekkoTokenizer: tokenizer + saved limits + default layout
ex = encode_choice(tok, state_json, instructions, options).with_target(soft_target)   # cacheable
batch = collate(examples, tasks=model.tasks, pad_multiple=64)
step = make_train_step(model, make_optimizer(total_steps=n_steps))   # v0 recipe: AdamW, clip 1.0, warmup+cosine
loss = step(batch)
save(model, "checkpoints/bekko-ft")              # loadable by the published PyTorch runtime
```

## API

| Function | Purpose |
|---|---|
| `load(path_or_repo=REPO_ID, revision=REVISION, *, dtype=mx.float32) -> BekkoMLX` | Loads published safetensors with a strict name/shape match. Also accepts a local export (for example one written by `save`). Attaches `model.tokenizer`. |
| `load_tokenizer(path_or_repo, revision) -> BekkoTokenizer` | Tokenizer, limits and default layout only, with no weights. |
| `encode_choice(tok, state_json, instructions, options, layout=None, **kw) -> EncodedDecision` | Token ids and branch structure, identical to `BekkoInference.prepare_groups`. `encode_decision(..., task="noul"/"score", values=...)` covers the other heads. |
| `EncodedDecision.to_dict/from_dict`, `dump_encoded(path, xs, tok)`, `load_encoded(path, tok)` | JSONL cache of encodings. The header holds a tokenizer+limits fingerprint, and a mismatched cache is refused. |
| `collate(examples, *, tasks=model.tasks, pad_multiple=1, max_options=None) -> DecisionBatch` | Pads prefixes to P and packs each example's option branches into one row of width T. Mixed tasks are allowed. |
| `choice_logits(model, batch) -> mx.array[B, K]` | Differentiable fp32 logits from each example's task head. Padded options are `NEG = -1e9`, and `batch.option_mask` marks the real ones. |
| `choice_probs(model, tok, state_json, instructions, options, *, layout=None) -> dict` | Probability per option key. `tok=None` uses `model.tokenizer`. |
| `noul_probability`, `score_prediction`, `decision_probs` | Typed helpers that mirror the runtime's `probability_yes`, `score` and `normalized_score`. |
| `soft_cross_entropy(logits, targets)`, `loss_fn(model, batch)`, `kl_divergence` | The trainer's loss: fp32 groupwise cross entropy against soft targets, averaged over decisions. Its gradient equals the KL gradient. |
| `make_optimizer(lr=1e-4, head_lr=5e-4, wd=0.01, *, total_steps=None)`, `recipe_schedule` | The v0 optimizer recipe (see "Training recipe" below). |
| `make_train_step(model, opt, *, max_grad_norm=1.0, compile=False)` | Runs value_and_grad, clips the global norm, and updates. Use `compile=True` together with `pad_multiple` bucketing. |
| `save(model, dir)` | Writes a complete SentenceTransformers export: the runtime files, both configs, the tokenizer, and fp32 `model.safetensors` under the original parameter names. |
| `random_model(seed, ...)` | A tiny random model with the v0 layout, for weight-free tests. |

## Architecture (exact values from the checkpoint)

`backbone_config.json` describes a ModernBERT (Ettin) backbone. It was initialised from
`cross-encoder/ettin-reranker-17m-v1` and fully fine-tuned by the v0 recipe. The model has 16,798,211 parameters, of which
12,894,208 are the embedding table.

| Field | Value | MLX implementation |
|---|---|---|
| vocab / hidden / heads x head_dim | 50,368 / 256 / 4 x 64 | `nn.Embedding`, `Linear` |
| layers / intermediate | 7 / 384 | GeGLU MLP: `Wi` 256->768, chunked into (a, gate), then `Wo(gelu(a) * gate)` with exact erf GELU |
| layer types | `F S S F S S F` (`global_attn_every_n_layers=3`) | one boolean mask per kind |
| sliding window | `local_attention=128`, so a token reads keys with \|q_pos - k_pos\| <= 64 | transformers sets `sliding_window=65` and the runtime uses `window=sliding_window-1` |
| RoPE | default rope with theta 160,000 for both global and local layers (stock ModernBERT uses 10,000 locally), rotate-half form | numpy cos/sin tables in fp32. `inv_freq` uses float32 pow and is bit-identical to torch |
| norms | LayerNorm, eps 1e-5, no bias. Embedding norm after `tok_embeddings`. Layer 0 has `attn_norm = Identity`. `final_norm` at the end | `mx.fast.layer_norm` |
| biases / dropout | none in attention, MLP or norms; all dropout 0.0 | checked at load. Unsupported configs raise an error |
| max positions | 7,999 | |
| heads | `choice`, `noul`, `score`: each `Linear(256, 1)` with bias, run in fp32. No ChoiceInteraction, no task tokens | per-example head selected by `task_index` |
| calibration | none. `inference_config.json` has no temperature or calibration constant, so probabilities are the plain per-decision fp32 softmax of head logits | |

**Shared-prefix branches** (`_SDPAPrefix`):

- The prefix is the instruction plus the state.
- The prefix is encoded bidirectionally. It never reads option tokens.
- At every layer, each option token attends to that layer's prefix keys and values plus its own branch, within the window on sliding layers.
- Branches never attend to each other.
- Option positions continue after their prefix: token j of an option sits at position P + j.
- Each option is represented by the mean of its final-normed branch tokens, including its trailing `[SEP]`.

The MLX version packs the K branches of an example into one row. `doc_seg` records which option each token belongs to, so prefix keys and values are concatenated once per example instead of being copied K times. The final prefix layer update is skipped because no option ever reads it.

**Input construction** (`encoding.py` mirrors each runtime function by name):

- Prefix: `[CLS] system? "Instruction: "+instruction "\n" "State: "+state_json [SEP]`, or with state first for `state_instruction`. The marker strings and the fields are tokenised separately.
- Options: `"Candidate: {key}: {description}" [SEP]`.
- Descriptions and instructions that are not strings become sorted compact JSON.
- The state string is used verbatim. Noul wraps it as `{"noul":{"yes":..,"no":..},"state":..}`.
- Budgets follow adaptive-v1: context 7,999, query cap 7,997, candidate cap 3,800.
- The query and the longest candidate split the context in half (the candidate gets the odd token), then lend each other unused capacity.
- Balanced truncation shares the query budget between instruction and state, and cuts the tail of each field.
- Candidates are right-truncated and keep their `[SEP]`.

## Parity with the published runtime

The reference is the published `BekkoSentenceTransformer`, loaded through `decision_mlx.bekko_provider.build`, running on CPU in fp32 with SDPA. Tests live in `tests/unit/test_bekko_mlx_parity.py` (marked `integration`). The cases come from `reference.parity_cases`: 16 deterministic OpenTTD-flavoured requests.

| Check | Result (bar) |
|---|---|
| Token ids: prefix and every candidate, all 16 cases | identical |
| Choice, 12 requests (2-12 options; 103 to 7,964-token prefixes; both layouts; system prompt; structured instructions and descriptions; unicode; query-truncated 9k-token state; candidate truncated at 3,800) | max \|p_mlx - p_torch\| = **1.91e-6** (<= 1e-4); argmax equal in 12/12 |
| Noul (2) / Score (2) | **9.5e-7** / **2.5e-7** (<= 1e-4). Typed fields (`probability_yes`, `score`, `normalized_score`) also match |
| Batched vs single-example logits (published weights, 7 mixed requests, `pad_multiple=32`) | **3.1e-6** (<= 1e-5) |
| Training: loss and every parameter gradient vs PyTorch autograd through `BekkoInference.forward` (6 mixed choice/noul/score decisions, both layouts) | loss diff **1.5e-6**; max \|grad diff\| **1.0e-5** against a largest gradient entry of 1.9 (bar: 1e-4 x max) |
| Round trip: perturbed MLX weights -> `save` -> published runtime (strict `load_state_dict`) | **2.4e-6** (<= 1e-4). The perturbation moved probabilities by up to 5.7e-2, so the check is meaningful. MLX reload is bit-identical |
| bf16 (weights and compute) vs fp32, 16 cases | max **2.6e-2**, median 3.5e-3, no argmax flips. Approximate; not a parity claim |

Run `python -m rh17m.bekko_mlx.bench parity` to reprint the per-case table.

## Deviations and caveats

- **Kernel layout differs, the maths does not.** The runtime splits prefixes longer than 384 tokens on sliding layers into 256-token query blocks. MLX computes the same windowed attention with one masked SDPA. It also packs option branches per row and skips the final prefix update. Only floating-point summation order differs (about 1e-6).
- **bf16.** There is no CUDA bf16 reference to compare against (the runtime's CPU path is fp32). The runtime itself warns that BF16 can move probabilities and occasionally the selected candidate.
  - `load(dtype=mx.bfloat16)` stores bf16 weights. Use it for inference only, because AdamW steps of about 1e-4 fall below bf16 resolution.
  - `model.set_compute_dtype(mx.bfloat16)` keeps fp32 master weights with bf16 activations (the `bf16` training mode). Pooling, heads, softmax and loss always run in fp32.
- **Not implemented.** Checkpoints with a ChoiceInteraction block, non-default RoPE, biases or dropout raise at `load`. `kind="ranking"` requests (Document candidates) raise at encode time. Task-marker tokens and `query_truncation="right"` are implemented but are not exercised by this checkpoint.
- **Limits.** Prefix plus the longest candidate is at most 7,999 positions.
  - States longer than about 7.9k tokens lose their tail.
  - When instruction and state both overflow, each keeps half the budget.
  - Put decision-critical fields early in the JSON.
  - The training recipe used query and document caps of 4,096 and 2,048, but the export (and `encode_*` by default) uses 7,997 and 3,800. For Railhead states of 200-1,000 tokens neither cap binds. Pass `query_length=` or `document_length=` to change them.
- **Layout.** v0 trained on a 1:1 mix of `instruction_state` and `state_instruction`. The export defaults to `instruction_state`. Fine-tune on whichever layout you serve, or on both.
- **Tokenizer.** `tokenizer.json` ships with a default truncation of 2,047 that HF overrides on every call. `BekkoTokenizer` disables it and applies the runtime's own truncation. Encodings depend on the tokenizer and the limits, and `dump_encoded` records a fingerprint of both.
- **Head biases.** Under the groupwise softmax loss, the gradient of a head bias is exactly zero (shift invariance). In training, only weight decay moves the biases, and that holds in PyTorch too. They cannot change any probability.
- **Saved exports** load with `BekkoSentenceTransformer(dir, trust_remote_code=True)` and with `rh17m.bekko_mlx.load(dir)`. `decision_mlx.bekko_provider.build` only accepts the pinned Hub release profiles, so it cannot load a local fine-tune.
- **Compilation.** `mx.compile` traces once per input shape. Bucket shapes with `collate(..., pad_multiple=64)`; padding never changes results.

## Training recipe (from `bekko_system_one.training`, `configs/bekko-system-one-v0-17m.yaml`)

- **Optimizer.** AdamW with weight decay 0.01 on every parameter. The encoder learning rate is 1e-4 and the head learning rate is 5e-4.
- **Clipping and schedule.** The global gradient norm is clipped at 1.0. The schedule is cosine with 10% linear warmup.
- **Batch size.** The logical batch is 512 decisions.
- **Loss.** `-(t * log_softmax(logits)).sum() / n_decisions`. Padded options carry a -1e9 logit.

`make_optimizer(total_steps=...)` and `make_train_step()` reproduce this. Without warmup, the first full-size Adam step visibly raises the loss on a toy batch; the integration test logs this.

## Performance on this Mac (Apple M4 Max, 36 GB, MLX 0.32.4)

**Measured under heavy contention.** During both benchmark runs, about 20 OpenTTD labelling servers were running, and the load average was 20-95 on 14 cores. These processes leave the GPU alone but starve MLX's host-side command encoding: step CPU time was about 70 ms against 290-1,500 ms of wall time, and identical runs varied by up to 3×. Treat every number below as a lower bound. Rerun on an idle machine with `python -m rh17m.bekko_mlx.bench all`, which prints JSON lines with the load average on each row.

The workload is synthetic OpenTTD-style compact JSON (`synthetic.py`). Each request has 9 options of about 33 tokens each, 300 option tokens in total. The bf16 training rows are mixed precision (fp32 master weights).

### Single choice latency (one request, 9 options, 50 warm runs)

| State | Mode | encode p50 / p90 | forward p50 / p90 | total p50 / p90 |
|---|---|---|---|---|
| 850-token prefix | fp32 eager | 1.0 / 1.2 ms | 8.3 / 10.3 ms | **9.3 / 11.3 ms** |
| 850-token prefix | fp32 `mx.compile` | 1.0 / 1.1 ms | 6.8 / 8.2 ms | **7.9 / 9.3 ms** |
| 850-token prefix | bf16 weights, compiled | 1.5 / 1.7 ms | 6.4 / 10.6 ms | 8.2 / 12.0 ms |
| 337-token prefix (about Railhead's 600-900-char states) | fp32 eager | 0.7 / 0.8 ms | 4.7 / 5.3 ms | **5.4 / 6.0 ms** |
| 337-token prefix | fp32 compiled | 0.7 / 0.9 ms | 5.4 / 9.2 ms | 6.1 / 10.0 ms |
| 850-token prefix, published PyTorch runtime, CPU fp32, 10 threads | `predict()` | | | 77 / 92 ms |

- "forward" covers collate, the encoder, the heads, the softmax and the host copy.
- At this size a single request is bound by dispatch and launch overhead. `mx.compile` saves about 1.5 ms at 850 tokens.
- bf16 does not help latency: on M4 Max, bf16 matmul is not faster than fp32 at these shapes.

### Training throughput (full fine-tune: forward, backward, clip, AdamW; 9 options; median of 10 steps)

| Prefix (padded width) | Batch | fp32 eager | fp32 compiled | bf16 eager | bf16 compiled | Peak memory (fp32 / bf16) |
|---|---|---|---|---|---|---|
| about 850 tokens (896) | 16 | 23.7 ex/s | 34.0 ex/s | 47.2 ex/s | **53.6 ex/s** | 5.3 / 2.9 GB |
| about 850 tokens (896) | 32 | 28.8 ex/s | 37.6 ex/s | 41.3 ex/s | **55.4 ex/s** | 9.4 / 5.0 GB |
| about 340 tokens (384) | 32 | 60.6 ex/s | 75.3 ex/s | 75.1 ex/s | **115.7 ex/s** | 5.1 / 2.8 GB |

- Batched inference at batch 32, 850 tokens, fp32 eager runs at 206 decisions/s.
- Work per example: about 13-15 GFLOP forward at 850 tokens, mostly projections and MLP. Attention is about a third of it.
- An earlier lower-load micro-measurement put the batched forward at about 3.5 ms/example.
- A blocked local-attention variant for the 4 sliding layers would save only about 5% of a training step, so it was not adopted.
- For Railhead-sized states (200-350 tokens), use `make_train_step(..., compile=True)` with `set_compute_dtype(mx.bfloat16)` and `pad_multiple=64`. Expect at least about 100 examples/s at batch 32 on a loaded machine.
