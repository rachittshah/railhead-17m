"""Latency and training-throughput benchmarks for the MLX Bekko port (numbers in docs/bekko-mlx.md).

    python -m rh17m.bekko_mlx.bench latency --state-tokens 800 --options 9 --runs 50 [--compile]
    python -m rh17m.bekko_mlx.bench throughput --batch-size 32 --dtype bf16 [--compile]
    python -m rh17m.bekko_mlx.bench parity          # published PyTorch runtime vs MLX, JSON report
    python -m rh17m.bekko_mlx.bench all             # the full documented matrix, JSON lines
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import mlx.core as mx
import numpy as np

from . import synthetic
from .batch import collate, decision_probs
from .encoding import encode_choice
from .io import load
from .train import make_optimizer, make_train_step

# name -> (parameter dtype, compute dtype). "bf16" is mixed precision (fp32 master weights, bf16 activations),
# the training mode to use; "bf16-params" stores bf16 weights (inference only: AdamW steps of ~1e-4 vanish
# below bf16 resolution).
DTYPES = {"fp32": (mx.float32, mx.float32), "bf16": (mx.float32, mx.bfloat16), "bf16-params": (mx.bfloat16, mx.bfloat16)}


def _percentiles(samples: list[float]) -> dict:
    ms = np.array(samples) * 1e3
    return {"p50_ms": round(float(np.percentile(ms, 50)), 3), "p90_ms": round(float(np.percentile(ms, 90)), 3)}


def _peak_memory_gb():
    getter = getattr(mx, "get_peak_memory", None)
    return None if getter is None else round(getter() / 2**30, 2)


def _reset_peak_memory():
    reset = getattr(mx, "reset_peak_memory", None)
    if reset is not None:
        reset()


def _model(dtype: str):
    params, compute = DTYPES[dtype]
    return load(dtype=params).set_compute_dtype(compute)


def choice_latency(model=None, *, state_tokens: int = 800, options: int = 9, runs: int = 50, warmup: int = 5,
                   compiled: bool = False, seed: int = 0) -> dict:
    """Single choice request: tokenize+encode, forward (collate, encoder, heads, softmax, host copy), total."""
    model = model or load()
    tok = model.tokenizer
    state, choices = synthetic.state_json(tok, state_tokens, seed), synthetic.options(options, seed)

    def forward_fn():
        if not compiled:
            return lambda batch: mx.softmax(model(batch), axis=-1)
        fn = mx.compile(lambda *arrays: mx.softmax(model.logits_from_arrays(*arrays), axis=-1), inputs=model.state)
        return lambda batch: fn(*batch.arrays())

    forward = forward_fn()
    encode_s, forward_s, total_s = [], [], []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        encoded = encode_choice(tok, state, synthetic.INSTRUCTIONS, choices)
        t1 = time.perf_counter()
        probabilities = np.array(forward(collate([encoded], tasks=model.tasks)))
        t2 = time.perf_counter()
        assert abs(probabilities.sum() - 1) < 1e-4
        if i >= warmup:
            encode_s.append(t1 - t0)
            forward_s.append(t2 - t1)
            total_s.append(t2 - t0)
    return {"bench": "choice_latency", "compiled": compiled, "dtype": str(model.compute_dtype),
            "prefix_tokens": len(encoded.prefix_ids), "options": options,
            "option_tokens": sum(map(len, encoded.option_ids)), "runs": runs,
            "encode": _percentiles(encode_s), "forward": _percentiles(forward_s), "total": _percentiles(total_s)}


def torch_latency(*, state_tokens: int = 800, options: int = 9, runs: int = 50, warmup: int = 5, seed: int = 0) -> dict:
    """The published runtime's predict() on CPU fp32 for the same request (reference point only)."""
    import torch

    from .encoding import native_request
    from .io import load_tokenizer
    from .reference import load_reference

    tok = load_tokenizer()
    request = native_request(synthetic.state_json(tok, state_tokens, seed), synthetic.INSTRUCTIONS,
                             synthetic.options(options, seed))
    ref = load_reference()
    samples = []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        ref.predict(request, show_progress_bar=False)
        if i >= warmup:
            samples.append(time.perf_counter() - t0)
    return {"bench": "torch_cpu_latency", "torch_threads": torch.get_num_threads(), "runs": runs,
            "total": _percentiles(samples)}


def _training_examples(tok, batch_size: int, state_tokens: int, options: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(batch_size):
        choices = synthetic.options(options, seed + i)
        target = rng.dirichlet(np.full(options, 0.5))
        examples.append(encode_choice(tok, synthetic.state_json(tok, state_tokens, seed + i), synthetic.INSTRUCTIONS,
                                      choices).with_target(target / target.sum()))
    return examples


def train_throughput(*, batch_size: int = 16, state_tokens: int = 800, options: int = 9, dtype: str = "fp32",
                     steps: int = 10, warmup: int = 3, compiled: bool = False, pad_multiple: int = 64) -> dict:
    """Full fine-tuning steps (forward, backward, AdamW) on distinct synthetic examples; examples/s."""
    model = _model(dtype)
    batch = collate(_training_examples(model.tokenizer, batch_size, state_tokens, options),
                    tasks=model.tasks, pad_multiple=pad_multiple)
    step = make_train_step(model, make_optimizer(), compile=compiled)
    for _ in range(warmup):
        step(batch)
    _reset_peak_memory()
    times, losses = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        losses.append(float(step(batch)))
        times.append(time.perf_counter() - t0)
    median = statistics.median(times)
    return {"bench": "train_throughput", "dtype": dtype, "compiled": compiled, "batch_size": batch_size,
            "prefix_width": batch.prefix_ids.shape[1], "option_width": batch.doc_ids.shape[1], "options": options,
            "step_ms_median": round(median * 1e3, 1), "examples_per_s": round(batch_size / median, 1),
            "peak_memory_gb": _peak_memory_gb(), "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)]}


def score_throughput(*, batch_size: int = 32, state_tokens: int = 800, options: int = 9, dtype: str = "fp32",
                     runs: int = 10, warmup: int = 3) -> dict:
    """Batched inference (no gradients) examples/s, e.g. for evaluation sweeps."""
    model = _model(dtype)
    examples = _training_examples(model.tokenizer, batch_size, state_tokens, options)
    for _ in range(warmup):
        decision_probs(model, examples, pad_multiple=64)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        decision_probs(model, examples, pad_multiple=64)
        times.append(time.perf_counter() - t0)
    return {"bench": "score_throughput", "dtype": dtype, "batch_size": batch_size,
            "examples_per_s": round(batch_size / statistics.median(times), 1)}


def parity_report() -> dict:
    from .reference import load_reference, parity, parity_cases

    model, ref = load(), load_reference()
    report = parity(model, ref, parity_cases(model.tokenizer))
    rows = [{k: row[k] for k in ("case", "task", "options", "prefix_tokens", "tokens_equal", "max_abs_diff",
                                 "argmax_equal")} for row in report.rows]
    return {"bench": "parity", "max_abs_diff": report.max_abs_diff, "by_task": report.by_task(), "rows": rows}


def environment() -> dict:
    """Host facts, including load average: concurrent CPU-heavy jobs starve MLX's command encoding."""
    return {"bench": "environment", "machine": platform.machine(), "platform": platform.platform(),
            "mlx": mx.__version__, "device": str(mx.default_device()), "python": platform.python_version(),
            "cpus": os.cpu_count(), "load_average": [round(x, 1) for x in os.getloadavg()]}


def run_all(emit=None) -> list[dict]:
    """The documented matrix. Each result is passed to ``emit`` as soon as it exists (streaming)."""
    out = []

    def record(result):
        result["load_average"] = [round(x, 1) for x in os.getloadavg()]
        out.append(result)
        if emit:
            emit(result)

    record(environment())
    for tokens in (800, 300):
        for compiled in (False, True):
            record(choice_latency(state_tokens=tokens, compiled=compiled))
    record(choice_latency(_model("bf16-params"), compiled=True))
    try:
        record(torch_latency())
    except ImportError:
        pass
    for tokens, sizes in ((800, (16, 32)), (300, (32,))):
        for batch_size in sizes:
            for dtype in ("fp32", "bf16"):
                for compiled in (False, True):
                    record(train_throughput(batch_size=batch_size, state_tokens=tokens, dtype=dtype, compiled=compiled))
                    mx.clear_cache()
    record(score_throughput(batch_size=32))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rh17m.bekko_mlx.bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["latency", "torch-latency", "throughput", "score", "parity", "all"])
    parser.add_argument("--state-tokens", type=int, default=800)
    parser.add_argument("--options", type=int, default=9)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--compile", action="store_true")
    a = parser.parse_args(argv)
    if a.what == "latency":
        results = [choice_latency(_model(a.dtype), state_tokens=a.state_tokens, options=a.options, runs=a.runs,
                                  compiled=a.compile)]
    elif a.what == "torch-latency":
        results = [torch_latency(state_tokens=a.state_tokens, options=a.options, runs=a.runs)]
    elif a.what == "throughput":
        results = [train_throughput(batch_size=a.batch_size, state_tokens=a.state_tokens, options=a.options,
                                    dtype=a.dtype, compiled=a.compile)]
    elif a.what == "score":
        results = [score_throughput(batch_size=a.batch_size, state_tokens=a.state_tokens, options=a.options,
                                    dtype=a.dtype)]
    elif a.what == "parity":
        results = [parity_report()]
    else:
        run_all(emit=lambda result: print(json.dumps(result), flush=True))
        return 0
    for result in results:
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
