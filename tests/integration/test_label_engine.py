"""End-to-end label engine on a real task start: needs the patched OpenTTD engine (Railhead checkout)."""

import pytest

from rh17m.dataset import DatasetConfig, build_example
from rh17m.labels import HOLD, LabelConfig, label_state
from rh17m.states import start_state

pytestmark = pytest.mark.integration


def test_start_state_labels_every_option_with_matched_seeds(tmp_path):
    meta = start_state("arena-train-0201", tmp_path)
    rec = label_state(tmp_path, meta.state_id, LabelConfig(seeds=2, train_seeds=2, chunks=2, max_air=3))
    assert rec["options"][-1] == HOLD and len(rec["options"]) >= 2
    hold = rec["outcomes"][HOLD]["rollouts"]
    assert len(hold) == 2 and all(len(r["nw"]) == 2 for r in hold)
    assert all("plane" in c for c in rec["candidates"])
    for key in rec["options"]:
        assert len(rec["outcomes"][key]["rollouts"]) == 2
    ex = build_example(rec, DatasetConfig())
    if ex is not None:  # a start state usually has a route that beats holding
        assert abs(sum(ex["target"].values()) - 1) < 1e-9
        assert ex["split"] == "train"
