"""Bekko v0 input construction, token for token identical to the published runtime.

Each function names the ``inference_v0.py`` function it mirrors at the pinned revision. A decision is one
shared prefix (``[CLS] system? Instruction: ... \\n State: ... [SEP]``, or state first) plus one branch per
option (``Candidate: <key>: <description> [SEP]``). Budgets follow adaptive-v1: the query and the longest
candidate split the context in half (candidate gets the odd token), then lend unused capacity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from tokenizers import Tokenizer

from .config import LAYOUTS, TASKS, InferenceSettings


def render_value(value) -> str:
    """inference_v0._text: strings unchanged, other JSON values as sorted compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def allocate_query_budget(first: int, second: int, budget: int) -> tuple[int, int]:
    """inference_v0.allocate_query_budget: half each (first wins the odd token), then lend unused room."""
    if min(first, second, budget) < 0:
        raise ValueError("lengths and budget must be nonnegative")
    a = min(first, (budget + 1) // 2)
    b = min(second, budget // 2)
    a += min(first - a, budget - a - b)
    b += min(second - b, budget - a - b)
    return a, b


@dataclass(frozen=True)
class QueryParts:
    """inference_v0.QueryParts: explicit instruction/state boundaries, never inferred from text."""

    instruction: str
    context: str
    system: str = ""
    layout: str = "instruction_state"

    def __post_init__(self):
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be nonempty text")
        if not isinstance(self.context, str) or not isinstance(self.system, str):
            raise TypeError("context and system must be text")
        if self.layout not in LAYOUTS:
            raise ValueError(f"unsupported prefix layout {self.layout!r}")

    def render(self) -> str:
        prefix = f"{self.system}\n\n" if self.system.strip() else ""
        instruction, context = f"Instruction: {self.instruction}", f"State: {self.context}"
        body = (instruction, context) if self.layout == "instruction_state" else (context, instruction)
        return prefix + "\n".join(body)


@dataclass(frozen=True)
class EncodedDecision:
    """One prefix plus K option branches, ready for :func:`rh17m.bekko_mlx.batch.collate`.

    ``target`` is an optional training distribution over options (nonnegative, sums to 1).
    ``values`` holds Score levels (output metadata only; never tokenized).
    """

    task: str
    keys: tuple[str, ...]
    prefix_ids: tuple[int, ...]
    option_ids: tuple[tuple[int, ...], ...]
    values: tuple[float, ...] | None = None
    target: tuple[float, ...] | None = None
    layout: str | None = None  # provenance only; the layout is already baked into prefix_ids

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}")
        if not self.option_ids or len(self.option_ids) != len(self.keys):
            raise ValueError("options and keys must align and be nonempty")
        if any(not ids for ids in self.option_ids) or len(self.prefix_ids) < 3:
            raise ValueError("every branch needs tokens")
        if self.target is not None:
            _check_distribution(self.target, len(self.keys))

    def with_target(self, target: Sequence[float]) -> EncodedDecision:
        return replace(self, target=tuple(float(t) for t in target))

    def to_dict(self) -> dict:
        """JSON-ready form (token ids and branch structure) for caching encodings to disk."""
        return {"task": self.task, "keys": list(self.keys), "prefix_ids": list(self.prefix_ids),
                "option_ids": [list(ids) for ids in self.option_ids],
                "values": None if self.values is None else list(self.values),
                "target": None if self.target is None else list(self.target), "layout": self.layout}

    @classmethod
    def from_dict(cls, row: dict) -> EncodedDecision:
        optional = lambda key: None if row.get(key) is None else tuple(row[key])
        return cls(task=row["task"], keys=tuple(row["keys"]), prefix_ids=tuple(row["prefix_ids"]),
                   option_ids=tuple(tuple(ids) for ids in row["option_ids"]), values=optional("values"),
                   target=optional("target"), layout=row.get("layout"))


EncodedChoice = EncodedDecision
ENCODED_FORMAT = "rh17m.bekko_mlx.encoded/v1"


def dump_encoded(path: str | Path, examples: Iterable[EncodedDecision], tok: BekkoTokenizer | None = None) -> int:
    """Write encodings as JSONL behind a header carrying the tokenizer/limits fingerprint. Returns the count."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w") as handle:
        handle.write(json.dumps({"format": ENCODED_FORMAT, "tokenizer": tok.fingerprint if tok else None}) + "\n")
        for example in examples:
            handle.write(json.dumps(example.to_dict(), separators=(",", ":")) + "\n")
            count += 1
    temporary.replace(path)
    return count


def load_encoded(path: str | Path, tok: BekkoTokenizer | None = None) -> list[EncodedDecision]:
    """Read :func:`dump_encoded` output; with ``tok``, refuse encodings made under another tokenizer/limits."""
    with Path(path).open() as handle:
        header = json.loads(handle.readline())
        if header.get("format") != ENCODED_FORMAT:
            raise ValueError("not an rh17m.bekko_mlx encoding file")
        if tok is not None and header.get("tokenizer") != tok.fingerprint:
            raise ValueError("encodings were produced with a different tokenizer or limits")
        return [EncodedDecision.from_dict(json.loads(line)) for line in handle if line.strip()]


def _check_distribution(target, count):
    if (len(target) != count or any(not math.isfinite(t) or t < 0 for t in target)
            or abs(sum(target) - 1) > 1e-5):
        raise ValueError("target must be a normalized distribution aligned with the options")


class BekkoTokenizer:
    """The checkpoint tokenizer plus the runtime's saved limits, truncation mode and default layout."""

    def __init__(self, tokenizer_dir: Path, settings: InferenceSettings, max_positions: int):
        tokenizer_dir = Path(tokenizer_dir)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        # tokenizer.json carries a default truncation (2047) that HF overrides per call; never apply it.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        special = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
        ids = {name: self.tokenizer.token_to_id(special[f"{name}_token"]) for name in ("cls", "sep", "pad")}
        if None in ids.values():
            raise ValueError("tokenizer lacks CLS/SEP/PAD tokens")
        if special.get("truncation_side", "right") != "right":
            raise ValueError("balanced query truncation requires a right-truncating tokenizer")
        self.cls_id, self.sep_id, self.pad_id = ids["cls"], ids["sep"], ids["pad"]
        self.settings = settings
        self.max_positions = max_positions
        self.task_token_ids = {}
        for task, marker in settings.task_tokens.items():
            token = self.tokenizer.token_to_id(marker)
            if token is None:
                raise ValueError("checkpoint tokenizer is missing a task token")
            self.task_token_ids[task] = token
        self._markers = self.ids(["Instruction: ", "State: ", "\n"])
        digest = hashlib.sha256((tokenizer_dir / "tokenizer.json").read_bytes())
        digest.update(json.dumps([settings.query_length, settings.document_length, settings.context_length,
                                  settings.query_truncation, settings.task_tokens]).encode())
        self.fingerprint = digest.hexdigest()[:16]  # tokenizer + limits; identifies cached encodings

    @property
    def prefix_layout(self) -> str:
        return self.settings.prefix_layout

    def ids(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tokenizer.encode_batch(texts, add_special_tokens=False)]

    def limits(self, query_length=None, document_length=None, context_length=None):
        """BekkoInference._limits: per-call caps, clamping saved defaults to a smaller context."""
        capacity = self.settings.context_length if context_length is None else context_length
        if not 5 <= capacity <= self.max_positions:
            raise ValueError("context length exceeds the backbone positional capacity")
        q = min(self.settings.query_length, capacity - 2) if query_length is None else query_length
        d = min(self.settings.document_length, capacity - 3) if document_length is None else document_length
        if not 3 <= q <= capacity - 2 or not 2 <= d <= capacity - 3:
            raise ValueError("invalid query/document limits for this backbone")
        return q, d, capacity

    def query_ids(self, parts: QueryParts, limit: int) -> list[int]:
        """balanced_query_ids (or right truncation of the rendered query) for one query."""
        if self.settings.query_truncation == "right":
            return [self.cls_id, *self.ids([parts.render()])[0][: limit - 2], self.sep_id]
        if limit < 3:
            raise ValueError("one valid query budget is required per query")
        system_text = f"{parts.system}\n\n" if parts.system.strip() else ""
        system, instruction, context = self.ids([system_text, parts.instruction, parts.context])
        instruction_marker, context_marker, separator = self._markers
        overhead = 2 + sum(map(len, self._markers))
        # The runtime pre-truncates each field to max(limits); allocation never reads past the budget, so
        # the untruncated lengths give the same split.
        budget = limit - overhead - len(system)
        if budget < 2:
            raise ValueError("system prompt and query markers leave fewer than two content tokens")
        ni, nc = allocate_query_budget(len(instruction), len(context), budget)
        ins = instruction_marker + instruction[:ni]
        ctx = context_marker + context[:nc]
        body = ins + separator + ctx if parts.layout == "instruction_state" else ctx + separator + ins
        return [self.cls_id, *system, *body, self.sep_id]

    def document_ids(self, documents: list[str], task: str, limit: int) -> list[list[int]]:
        """BekkoInference._documents: optional task marker, right truncation, final SEP."""
        encoded = [ids[: limit - 1] for ids in self.ids(documents)]
        if task in self.task_token_ids:
            return [[self.task_token_ids[task], *d[: limit - 2], self.sep_id] for d in encoded]
        return [[*d, self.sep_id] for d in encoded]


def _options(options: Iterable[tuple[str, object]]):
    options = [tuple(option) for option in options]
    if not options or any(len(option) != 2 for option in options):
        raise ValueError("options must be a nonempty list of (key, description) pairs")
    keys = tuple(key for key, _ in options)
    if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("option keys must be distinct nonempty strings")
    return keys, [render_value(description) for _, description in options]


def render_decision(state_json: str, instructions, options, *, task="choice", layout="instruction_state",
                    system: str = ""):
    """inference_v0.input_groups for one judgment: (QueryParts, candidate texts, keys)."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    if not isinstance(state_json, str):
        raise TypeError("state_json must be JSON text")
    state = json.loads(state_json)
    keys, descriptions = _options(options)
    context = state_json
    if task == "noul":
        if set(keys) not in ({"true", "false"}, {"yes", "no"}):
            raise ValueError("noul requires authored true/false or yes/no options")
        mapping = dict(zip(keys, descriptions, strict=True))
        yes, no = ("true", "false") if "true" in keys else ("yes", "no")
        context = json.dumps({"noul": {"yes": mapping[yes], "no": mapping[no]}, "state": state},
                             ensure_ascii=False, separators=(",", ":"))
    parts = QueryParts(render_value(instructions), context, system or "", layout)
    candidates = [f"Candidate: {key}: {description}" for key, description in zip(keys, descriptions, strict=True)]
    return parts, candidates, keys


def encode_decision(tok: BekkoTokenizer, state_json: str, instructions, options, *, task: str = "choice",
                    layout: str | None = None, system: str = "", values: Sequence[float] | None = None,
                    target: Sequence[float] | None = None, query_length: int | None = None,
                    document_length: int | None = None, context_length: int | None = None) -> EncodedDecision:
    """Token ids for one decision exactly as BekkoInference.prepare_groups builds them.

    ``options`` is a list of (key, description); descriptions may be strings or JSON values.
    ``values`` (Score only) are the numeric levels, never tokenized.
    """
    parts, candidates, keys = render_decision(state_json, instructions, options, task=task,
                                              layout=layout or tok.prefix_layout, system=system)
    if task == "score":
        if values is None or len(values) != len(keys) or len(keys) < 2:
            raise ValueError("score requires one numeric value per option and at least two options")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("score values must be finite numbers")
        if len(set(values)) != len(values):
            raise ValueError("score values must be distinct")
        values = tuple(float(v) for v in values)
    elif values is not None:
        raise ValueError("values apply only to score decisions")
    qlimit, dlimit, capacity = tok.limits(query_length, document_length, context_length)
    documents = tok.document_ids(candidates, task, dlimit)
    query = tok.query_ids(parts, qlimit)
    dsize, qsize = allocate_query_budget(max(map(len, documents)), len(query), capacity)
    if len(query) > qsize:
        query = tok.query_ids(parts, qsize)
    documents = [d[: dsize - 1] + [d[-1]] if len(d) > dsize else d for d in documents]
    return EncodedDecision(task=task, keys=keys, prefix_ids=tuple(query),
                           option_ids=tuple(tuple(d) for d in documents), values=values,
                           target=None if target is None else tuple(float(t) for t in target), layout=parts.layout)


def encode_choice(tok: BekkoTokenizer, state_json: str, instructions, options, layout: str | None = None,
                  **kwargs) -> EncodedDecision:
    """Encode one choice question: ``options`` is a list of (key, description)."""
    return encode_decision(tok, state_json, instructions, options, task="choice", layout=layout, **kwargs)


def native_request(state_json: str, instructions, options, *, task: str = "choice", decision_id: str = "decision",
                   values: Sequence[float] | None = None, system: str = "") -> dict:
    """The published runtime's native request for one decision.

    Mirrors decision_mlx.bekko_provider's typed-request translation (``_json`` fields are json.dumps of
    the authored value, ensure_ascii=False). Used to drive the PyTorch reference.
    """
    dumps = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
    options = [tuple(option) for option in options]
    return {"state_json": state_json, "decisions": [{
        "id": decision_id, "kind": "judgment", "type": task, "instructions_json": dumps(instructions),
        "system_prompt": system,
        "criteria": [{"id": key, "description_json": dumps(description),
                      "value": values[i] if task == "score" else None}
                     for i, (key, description) in enumerate(options)],
        "documents": [], "scoring": None}]}
