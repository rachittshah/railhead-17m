"""The published PyTorch runtime (CPU, fp32, SDPA) as the parity reference, plus the parity case suite.

Needs the ``reference`` extra (torch, transformers, sentence-transformers); imported lazily.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import synthetic
from .batch import decision_probs
from .config import REPO_ID, REVISION
from .encoding import encode_decision, native_request


def load_reference(path: str | Path | None = None, *, revision: str = REVISION):
    """Published BekkoSentenceTransformer: the pinned Hub release, or a local export (e.g. from save())."""
    if path is None:
        from decision_mlx.bekko_provider import build

        return build({"model_id": REPO_ID, "revision": revision, "trust_remote_code": True}).model
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = str(Path(path).resolve())
    model_class = get_class_from_dynamic_module("inference_v0.BekkoSentenceTransformer", path, local_files_only=True)
    return model_class(path, trust_remote_code=True, device="cpu", local_files_only=True, attn_implementation="sdpa")


@dataclass(frozen=True)
class Case:
    name: str
    state_json: str
    instructions: object
    options: list
    task: str = "choice"
    layout: str = "instruction_state"
    system: str = ""
    values: tuple | None = None

    def request(self) -> dict:
        return native_request(self.state_json, self.instructions, self.options, task=self.task,
                              values=self.values, system=self.system)

    def encode(self, tok):
        return encode_decision(tok, self.state_json, self.instructions, self.options, task=self.task,
                               layout=self.layout, system=self.system, values=self.values)


def reference_probs(ref, case: Case) -> dict[str, float]:
    result = ref.predict(case.request(), prefix_layout=case.layout, show_progress_bar=False)
    return result["decision"]["probabilities"]


def reference_encoding(ref, case: Case) -> tuple[list[int], list[list[int]]]:
    """Token ids the runtime feeds its encoder: (query, candidates) from BekkoInference.prepare_groups."""
    runtime = importlib.import_module(type(ref[0]).__module__)
    (group,) = runtime.input_groups(case.request(), prefix_layout=case.layout)
    (prepared,) = ref[0].prepare_groups([group])
    return list(prepared.query), [list(d) for d in prepared.documents]


def parity_cases(tok) -> list[Case]:
    """12 choice + 2 noul + 2 score requests: 2-12 options, ~60 to ~9,000-token states, both layouts,
    a system prompt, structured instructions/descriptions, unicode, and both truncation paths."""
    S, O = synthetic.state_json, synthetic.options
    ins = synthetic.INSTRUCTIONS
    huge = [("route_1", "Build a coal route. " + " ".join(synthetic.TOWNS) * 330), ("hold", "Keep the cash.")]
    return [
        Case("short-2", S(tok, 60, 1), ins, O(2, 1)),
        Case("short-3-state-first", S(tok, 60, 2), ins, O(3, 2), layout="state_instruction"),
        Case("800-9", S(tok, 800, 3), ins, O(9, 3)),
        Case("800-9-structured-state-first", S(tok, 800, 4), ins, O(9, 4, structured=True), layout="state_instruction"),
        Case("800-12-system", S(tok, 800, 5), ins, O(12, 5), system="You are the dispatcher of a small railway."),
        Case("400-5", S(tok, 400, 6), "Quel itinéraire choisir ? Réponds pour la compagnie.", O(5, 6)),
        Case("1000-7-structured", S(tok, 1000, 7), ins, O(7, 7, structured=True)),
        Case("200-4", S(tok, 200, 8), ins, O(4, 8)),
        Case("800-6-long-options", S(tok, 800, 9), ins, O(6, 9, detail=120)),
        Case("9000-3-query-truncated", S(tok, 9000, 10), ins, O(3, 10)),
        Case("short-2-candidate-truncated", S(tok, 60, 11), ins, huge),
        Case("600-10-structured-instructions", S(tok, 600, 12),
             {"goal": "maximise company value", "horizon_months": 12, "constraints": ["no new loans"]}, O(10, 12)),
        Case("noul-800", S(tok, 800, 13), "Should the company take an extra loan this month?",
             [("false", "No: current cash covers planned builds."), ("true", "Yes: borrowing unlocks a profitable route.")],
             task="noul"),
        Case("noul-yes-no-state-first", S(tok, 300, 14), "Is any route losing money?",
             [("yes", "At least one route had negative profit last year."), ("no", "Every route was profitable.")],
             task="noul", layout="state_instruction"),
        Case("score-5", S(tok, 800, 15), "How urgent is expanding the network?",
             [(str(i), d) for i, d in enumerate(["not at all", "slightly", "moderately", "very", "critical"])],
             task="score", values=(0, 1, 2, 3, 4)),
        Case("score-3-uneven", S(tok, 150, 16), "Expected change in company rating next year?",
             [("down", "it falls"), ("flat", "it stays the same"), ("up", "it rises")], task="score",
             values=(-1.0, 0.5, 2.0)),
    ]


@dataclass
class ParityReport:
    rows: list[dict] = field(default_factory=list)

    @property
    def max_abs_diff(self) -> float:
        return max(row["max_abs_diff"] for row in self.rows)

    def by_task(self) -> dict[str, float]:
        out = {}
        for row in self.rows:
            out[row["task"]] = max(out.get(row["task"], 0.0), row["max_abs_diff"])
        return out


def parity(model, ref, cases: list[Case]) -> ParityReport:
    """Per case: token-id equality with the runtime, and max |p_mlx - p_torch| over options."""
    report = ParityReport()
    for case in cases:
        encoded = case.encode(model.tokenizer)
        query, documents = reference_encoding(ref, case)
        t0 = time.perf_counter()
        expected = reference_probs(ref, case)
        torch_s = time.perf_counter() - t0
        got = decision_probs(model, [encoded])[0]
        diff = max(abs(got[k] - expected[k]) for k in expected)
        report.rows.append({
            "case": case.name, "task": case.task, "options": len(case.options), "prefix_tokens": len(encoded.prefix_ids),
            "option_tokens": [len(d) for d in encoded.option_ids],
            "tokens_equal": list(encoded.prefix_ids) == query and [list(d) for d in encoded.option_ids] == documents,
            "max_abs_diff": float(diff), "argmax_equal": max(got, key=got.get) == max(expected, key=expected.get),
            "torch_s": round(torch_s, 3), "mlx": got, "torch": expected,
        })
    return report


def reference_loss_and_grads(ref, cases: list[Case], targets: list[list[float]]):
    """Trainer loss (bekko_system_one.loss.distribution_loss_sum / decisions) and its parameter gradients,
    by PyTorch autograd through the published BekkoInference.forward on runtime-collated features."""
    import torch

    runtime = importlib.import_module(type(ref[0]).__module__)
    module = ref[0].eval()
    groups = [runtime.input_groups(case.request(), prefix_layout=case.layout)[0] for case in cases]
    features = runtime.collate_groups(module.prepare_groups(groups), module)
    parameters = dict(module.named_parameters())
    for parameter in parameters.values():
        parameter.requires_grad_(True)
        parameter.grad = None
    with torch.enable_grad():
        scores = module(features)["scores"].flatten()
        logits = torch.nn.utils.rnn.pad_sequence(scores.split([len(t) for t in targets]), batch_first=True,
                                                 padding_value=-1e9).float()
        padded = torch.tensor([list(t) + [0.0] * (logits.shape[1] - len(t)) for t in targets])
        loss = -(padded * logits.log_softmax(-1)).sum() / len(targets)
        loss.backward()
    grads = {name: p.grad.detach().numpy().copy() for name, p in parameters.items() if p.grad is not None}
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    return loss.item(), grads


def max_diff(a: dict[str, float], b: dict[str, float]) -> float:
    return float(np.max([abs(a[k] - b[k]) for k in a]))
