"""End-to-end smoke: tiny build -> leakage audit -> heuristics -> eval -> mock VLM harness."""
import json

from relbind import build, eval as rb_eval
from relbind.baselines import heuristics
from relbind.vlm import run as vlm_run


def test_pipeline(tmp_path):
    out = tmp_path / "bench"
    build.main(["--config", "configs/prototype.json", "--out", str(out), "--limit", "1"])
    man = json.loads((out / "manifest.json").read_text())
    assert man["leakage_ok"]
    exs = [json.loads(l) for l in open(out / "test_id" / "examples.jsonl", encoding="utf-8")]
    assert {e["variant"] for e in exs} == {"orig", "cf", "reroute", "style"}
    # every flipping query has different gold answers on orig and cf
    by_q = {}
    for e in exs:
        by_q.setdefault(e["query_id"], {})[e["variant"]] = e["answer"]
    for qid, v in by_q.items():
        flips = next(e["flips_under_cf"] for e in exs if e["query_id"] == qid)
        assert (v["orig"] != v["cf"]) == flips
        assert v["orig"] == v["style"] == v["reroute"]

    preds = tmp_path / "prox.jsonl"
    heuristics.main(["--data", str(out), "--method", "proximity", "--out", str(preds)])
    res_path = tmp_path / "m.json"
    rb_eval.main(["--data", str(out), "--pred", str(preds), "--out", str(res_path), "--B", "50"])
    res = json.loads(res_path.read_text())
    for split, r in res["splits"].items():
        # proximity ignores connectors -> can never get both members of a swap pair right
        assert r["paired_cf_accuracy"]["value"] == 0.0, split

    run_dir = tmp_path / "mock_run"
    vlm_run.main(["--data", str(out), "--splits", "test_id", "--condition", "C_oracle_crop", "--backend", "mock",
                  "--mock-from", str(preds), "--out", str(run_dir)])
    got = [json.loads(l) for l in open(run_dir / "predictions.jsonl", encoding="utf-8")]
    ref = {json.loads(l)["example_id"]: json.loads(l)["prediction"] for l in open(preds, encoding="utf-8")}
    assert got and all(g["prediction"] == ref[g["example_id"]] for g in got)
