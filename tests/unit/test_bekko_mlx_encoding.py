"""Bekko MLX input construction rules, the encoding cache, and request translation (no weights needed).

Token-level equality with the published runtime is checked in test_bekko_mlx_parity.py.
"""

import json
from types import SimpleNamespace

import pytest

from rh17m.bekko_mlx.encoding import (
    EncodedDecision,
    QueryParts,
    allocate_query_budget,
    dump_encoded,
    load_encoded,
    native_request,
    render_value,
)


@pytest.mark.parametrize(("first", "second", "budget", "expected"), [
    (10, 10, 7, (4, 3)),  # odd token goes to the first field
    (2, 100, 10, (2, 8)),  # unused capacity is lent to the other field
    (100, 2, 10, (8, 2)),
    (3, 4, 20, (3, 4)),
    (0, 50, 9, (0, 9)),
])
def test_allocate_query_budget(first, second, budget, expected):
    assert allocate_query_budget(first, second, budget) == expected


def test_query_rendering_layouts_and_system_prompt():
    assert QueryParts("Pick", '{"a":1}').render() == 'Instruction: Pick\nState: {"a":1}'
    assert QueryParts("Pick", "{}", layout="state_instruction").render() == "State: {}\nInstruction: Pick"
    assert QueryParts("Pick", "{}", system="Be brief").render() == "Be brief\n\nInstruction: Pick\nState: {}"
    assert QueryParts("Pick", "{}", system="  ").render() == "Instruction: Pick\nState: {}"
    with pytest.raises(ValueError, match="instruction"):
        QueryParts(" ", "{}")


def test_render_value_matches_runtime_text():
    assert render_value("as authored ") == "as authored "
    assert render_value({"b": 1, "a": "é"}) == '{"a":"é","b":1}'


def test_encoded_decision_cache_round_trip(tmp_path):
    tok = SimpleNamespace(fingerprint="abc123")
    examples = [EncodedDecision("choice", ("r1", "hold"), (1, 5, 2), ((7, 2), (8, 9, 2)), target=(0.25, 0.75),
                                layout="instruction_state"),
                EncodedDecision("score", ("0", "1"), (1, 6, 2), ((3, 2), (4, 2)), values=(0.0, 2.0))]
    path = tmp_path / "enc.jsonl"
    assert dump_encoded(path, examples, tok) == 2
    assert load_encoded(path, tok) == examples
    with pytest.raises(ValueError, match="different tokenizer"):
        load_encoded(path, SimpleNamespace(fingerprint="other"))


def test_target_must_be_a_distribution():
    with pytest.raises(ValueError, match="normalized"):
        EncodedDecision("choice", ("a", "b"), (1, 5, 2), ((3, 2), (4, 2)), target=(0.5, 0.6))


def test_native_request_matches_decision_mlx_translation():
    """Our reference request equals what decision_mlx's Bekko provider sends the published runtime."""
    pytest.importorskip("torch")  # BekkoProvider's runtime wrapper imports torch
    from decision_mlx.bekko_provider import BekkoProvider

    sent = []

    class Recorder:
        def predict(self, request, show_progress_bar=False):
            sent.append(request)
            return {d["id"]: {"probabilities": {c["id"]: 1 / len(d["criteria"]) for c in d["criteria"]}}
                    for d in request["decisions"]}

    provider = BekkoProvider(Recorder(), metadata={"model_id": "bekko"}, device="cpu")
    state = {"towns": [{"name": "Bürgerstadt", "pop": 1200}], "money": 1.5}
    state_json = json.dumps(state, ensure_ascii=False)
    provider._call(state, {"d": {"type": "choice", "instructions": "Pick", "criteria": {"r1": "Build", "hold": {"x": 1}}}})
    assert sent[-1] == native_request(state_json, "Pick", [("r1", "Build"), ("hold", {"x": 1})], decision_id="d")
    provider._call(state, {"d": {"type": "noul", "instructions": {"q": "Loan?"}, "criteria": {"true": "yes", "false": "no"}}})
    assert sent[-1] == native_request(state_json, {"q": "Loan?"}, [("false", "no"), ("true", "yes")], task="noul",
                                      decision_id="d")
    provider._call(state, {"d": {"type": "score", "instructions": "Urgency", "criteria": ["low", "high"],
                                 "score_values": [1, 5]}})
    assert sent[-1] == native_request(state_json, "Urgency", [("0", "low"), ("1", "high")], task="score",
                                      values=[1, 5], decision_id="d")
