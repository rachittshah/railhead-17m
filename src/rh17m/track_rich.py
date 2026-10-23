"""What a run shows beyond scalar curves: a decisions table, decision traces and a Markdown report.

The table has one row per evaluated decision (the pick, the simulator's best option, regret, confidence). A
trace replays one decision the way a reviewer would read it: the question and options the model saw (user),
the model's pick with its probabilities (assistant), the simulator's values per option (metadata) and the
forward pass as a generation span. The report puts the model next to the scripted baselines.
"""

from __future__ import annotations

import json

DECISION_COLUMNS = ["id", "source", "options", "pick", "p_pick", "best", "regret", "capture", "hit"]


def _pick(p: dict[str, float]) -> str:
    return max(p, key=lambda k: (p[k], k == "hold"))


def decision_rows(examples: list[dict], probs: list[dict[str, float]]) -> list[list]:
    from rh17m.evaluate import decision_metrics

    rows = []
    for e, p in zip(examples, probs, strict=True):
        m = decision_metrics(e, p)
        pick = _pick(p)
        rows.append([e["id"], e["source"], len(e["options"]), pick, round(p[pick], 4), e["best"],
                     round(m["regret"], 4), round(m["capture"], 3), int(m["hit"])])
    return rows


def decision_trace(ex: dict, probs: dict[str, float], model: str, latency_ms: float | None = None,
                   state_chars: int = 1500) -> dict:
    """Keyword arguments for ``Tracker.log_trace``."""
    from rh17m.evaluate import decision_metrics

    pick = _pick(probs)
    m = decision_metrics(ex, probs)
    options = "\n".join(f"- {k}: {text}" for k, text in ex["options"])
    state = ex["state_json"] if len(ex["state_json"]) <= state_chars else ex["state_json"][:state_chars] + " ..."
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    answer = (f"Pick: {pick} (p={probs[pick]:.2f}). Simulator best: {ex['best']}; regret {m['regret']:.3f}.\n"
              + "\n".join(f"{k}: p={p:.3f}, value={ex['q'][k]:.3f}" for k, p in ranked))
    span = {"id": "forward", "name": "bekko choice head", "kind": "generation", "model": model,
            "input": {"options": len(ex["options"]), "state_chars": len(ex["state_json"])},
            "output": {"pick": pick, "p": round(probs[pick], 4)}}
    if latency_ms is not None:
        span["duration_ms"] = round(latency_ms, 2)
    return {"messages": [{"role": "user", "content": f"{ex['instructions']}\n\nState: {state}\n\nOptions:\n{options}"},
                         {"role": "assistant", "content": answer}],
            "metadata": {"state_id": ex["id"], "source": ex["source"], "best": ex["best"], "pick": pick,
                         "regret": round(m["regret"], 4), "hit": bool(m["hit"]),
                         "values": {k: round(v, 4) for k, v in ex["q"].items()},
                         "status": "ok" if m["hit"] else "miss",
                         **({"latency_ms": round(latency_ms, 2)} if latency_ms is not None else {})},
            "spans": [span]}


def report_markdown(results: dict, config: dict) -> str:
    summary = (f"Examples: {json.dumps(results['examples'])}; trained on {results['train_used']} "
               f"({results.get('distilled', 0)} teacher-labelled). Best epoch {results['best_epoch']}, "
               f"temperature {results['temperature']:.3f}, {results['train_s']} s.")
    lines = [f"# {results['name']}", "", summary, ""]
    for split in ("val", "test", "ood"):
        if split not in results:
            continue
        lines += [f"## {split} (n={int(results[split]['n'])})", "",
                  "| policy | regret | exp. regret | capture | hit |", "|---|---|---|---|---|"]
        rows = [("**railhead-17m**", results[split])]
        rows += [(name, m) for name, m in results.get(f"{split}_baselines", {}).items() if name != "oracle"]
        for name, m in rows:
            lines.append(f"| {name} | {m['regret']:.4f} | {m['exp_regret']:.4f} | {m['capture']:.3f} | {m['hit']:.3f} |")
        lines.append("")
    lines += ["## Config", "", "```json", json.dumps(config, indent=1), "```"]
    return "\n".join(lines)
