from types import SimpleNamespace

from rh17m import train


def test_encodings_are_shared_by_tokenizers_with_the_same_fingerprint(monkeypatch):
    """Every model load builds a new tokenizer object: the cache must key on what the tokenizer is (its
    fingerprint), or a long climb re-encodes everything per run and keeps every copy alive."""
    calls = []

    def fake_encode(tok, state_json, instructions, options, layout):
        calls.append(tok.fingerprint)
        return ("enc", tok.fingerprint, layout)

    monkeypatch.setattr(train.B, "encode_choice", fake_encode)
    monkeypatch.setattr(train, "_ENC_CACHE", {})
    ex = {"id": "s1", "state_json": "{}", "instructions": "q?", "options": [("a", "A"), ("hold", "H")]}
    first, second = SimpleNamespace(fingerprint="f1"), SimpleNamespace(fingerprint="f1")
    assert train._encoded(first, ex, "instruction_state") == train._encoded(second, ex, "instruction_state")
    assert calls == ["f1"]
    other = SimpleNamespace(fingerprint="f2")  # another tokenizer or limits: its own encodings
    assert train._encoded(other, ex, "instruction_state")[1] == "f2"
    assert calls == ["f1", "f2"]
