from pathlib import Path

from rh17m.climb import apply, coarse_moves, resolve_teacher
from rh17m.train import TrainConfig


def test_apply_merges_data_overrides_and_keeps_tuples():
    base = TrainConfig(data={"tau": 0.05})
    new = apply(base, {"data": {"ranks": True}, "layouts": ["instruction_state"]}, "x")
    assert new.data == {"tau": 0.05, "ranks": True}
    assert new.layouts == ("instruction_state",) and new.name == "x"
    assert base.data == {"tau": 0.05}  # the incumbent is not mutated


def test_teacher_references_resolve_or_skip(tmp_path):
    assert resolve_teacher("@incumbent", tmp_path) == str(tmp_path)
    assert resolve_teacher("@incumbent", None) is None
    assert resolve_teacher("bekko-68m", None) == "bekko-68m"
    assert apply(TrainConfig(), {"distill_from": "@incumbent"}, "y", None) is None
    assert apply(TrainConfig(), {"distill_from": "@incumbent"}, "y", Path("/m")).distill_from == "/m"


def test_every_coarse_move_names_a_real_setting():
    fields = set(TrainConfig.__dataclass_fields__) | {"_control"}
    for move in coarse_moves():
        assert set(move) <= fields, move


def test_control_move_changes_nothing():
    from rh17m.climb import CONTROL

    base = TrainConfig(data={"tau": 0.05}, lr=2e-4)
    same = apply(base, CONTROL, "c")
    assert same.data == base.data and same.lr == base.lr and same.name == "c"
