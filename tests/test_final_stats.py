"""Statistics + audit-scoring code tests on toy inputs (code tests only; no reviewer data is simulated)."""
import csv
import json

from relbind.final_analysis import paired_diff, rate
from relbind.metrics import design_effect, stratified_cluster_bootstrap
from relbind.validation import cohen_kappa, score2


def test_rate_counts_and_cluster_ci():
    items = [("g1", True), ("g1", True), ("g2", False), ("g2", False)]
    r = rate(items, B=500)
    assert (r["num"], r["den"], r["value"]) == (2, 4, 0.5)
    assert 0.0 <= r["ci95"][0] <= 0.5 <= r["ci95"][1] <= 1.0
    assert rate([])["value"] is None


def test_paired_diff_mcnemar():
    items = [(f"g{i}", True, False) for i in range(10)] + [(f"h{i}", True, True) for i in range(10)]
    d = paired_diff(items, B=300)
    assert d["diff_pp"] == 50.0 and d["discordant_a_only"] == 10 and d["discordant_b_only"] == 0
    assert d["mcnemar_p"] < 0.01


def test_design_effect_perfect_clustering():
    Q = []
    for g in range(20):
        for k in range(4):
            Q.append({"group_id": f"g{g}", "o": g % 2 == 0})
    de = design_effect(Q, lambda q: q["o"])
    assert de["icc"] > 0.95 and de["design_effect"] > 3.5


def test_stratified_bootstrap_keeps_point_estimate():
    Q = [{"group_id": f"g{i}", "template": "t1" if i < 5 else "t2", "v": i % 2} for i in range(10)]
    p, ci = stratified_cluster_bootstrap(Q, lambda qs: sum(q["v"] for q in qs) / len(qs), B=200)
    assert p == 0.5 and ci[0] <= p <= ci[1]


def test_kappa_and_score2(tmp_path):
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    key = [{"item": 0, "example_id": "x-orig", "query_id": "q", "answer": "C1", "variant": "orig"},
           {"item": 1, "example_id": "x-cf", "query_id": "q", "answer": "C2", "variant": "cf"}]
    kp = tmp_path / "key.json"
    kp.write_text(json.dumps(key))

    def sheet(name, rows):
        p = tmp_path / name
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "image", "question", "reviewer_answer", "unambiguous(y/n)", "notes"])
            for r in rows:
                w.writerow(r)
        return p

    a = sheet("a.csv", [[0, "", "", "C1", "y", ""], [1, "", "", "C2", "y", ""]])
    b = sheet("b.csv", [[1, "", "", "C2", "y", ""], [0, "", "", "C3", "n", ""]])
    r = score2(kp, a, b)
    assert r["reviewer_A"]["accuracy"] == 1.0 and r["reviewer_B"]["accuracy"] == 0.5
    assert r["pairs"]["both_members_correct_by_both_reviewers"] == 0.0
    assert not r["pass"] and len(r["failed_items"]) == 1
