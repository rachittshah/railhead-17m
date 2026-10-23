"""Model card for a training run, written from its results.json (numbers are never typed by hand)."""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = """---
license: mit
base_model: hotchpotch/bekko-system-one-v0-17m
library_name: mlx
tags:
- typed-decisions
- system-one
- openttd
- railhead
- mlx
---

# railhead-17m ({name})

A 17M-parameter System-1 decision model for [Railhead](https://huggingface.co/datasets/rachittshah/railhead)
(OpenTTD): given a company's wake-up state, it scores which air route to build now, or to hold. Fine-tuned
from Bekko System One v0 17M in native MLX on simulator counterfactual labels: every option was built in a
fork of the game and played forward for up to two years over matched seeds.

## Results (held-out maps)

Values are asinh(net worth / £100k) at the end of the rollout or the game; 0.1 is roughly 10% of net worth.
`regret` = best option's value minus the picked option's; `capture` = share of the best option's edge over a
random pick that the pick keeps (1 = oracle).

| split | n | model regret | model capture | model hit | longest-route regret | longest capture | uniform exp. regret |
|---|---|---|---|---|---|---|---|
{rows}

Temperature (fitted on val NLL): {temperature:.3f}. Best epoch: {best_epoch}.

## Use (MLX)

```python
from rh17m import bekko_mlx as B
model = B.load("<local snapshot of this repo>", revision=None)
probs = B.choice_probs(model, None, state_json, instructions, [("route_a", "..."), ("hold", "build nothing")])
```

The weights keep the published Bekko parameter names, so the published PyTorch runtime loads them too.

## Training

Config: `config.json`. Per-epoch history: `history.jsonl`. Code: private repo `rachittshah/railhead-17m`.
"""


def card(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "results.json").read_text())
    rows = []
    for split in ("val", "test", "ood"):
        if split not in res:
            continue
        m, b = res[split], res.get(f"{split}_baselines", {})
        lon, uni = b.get("longest", {}), b.get("uniform", {})
        rows.append(f"| {split} | {int(m['n'])} | {m['regret']:.4f} | {m['capture']:.3f} | {m['hit']:.3f} | "
                    f"{lon.get('regret', float('nan')):.4f} | {lon.get('capture', float('nan')):.3f} | "
                    f"{uni.get('exp_regret', float('nan')):.4f} |")
    return TEMPLATE.format(name=res["name"], rows="\n".join(rows), temperature=res["temperature"],
                           best_epoch=res["best_epoch"])
