from relbind.metrics import compare, is_correct, mcnemar_exact, metric_fns, normalize, query_table
from relbind.vlm.parse import parse


def _rows(pred_orig, pred_cf, n=4):
    rows = []
    for g in range(n):
        for var, ans, pred in (("orig", "C1", pred_orig), ("cf", "C2", pred_cf), ("style", "C1", pred_orig)):
            rows.append({"query_id": f"q{g}", "group_id": f"g{g}", "split": "s", "family": "schematic",
                         "template": "elevation", "topology": "straight", "query_type": "a2e",
                         "flips_under_cf": True, "answerable": True, "features": {}, "variant": var,
                         "answer": ans, "prediction": pred, "correct": is_correct(pred, ans)})
    return rows


def test_position_only_strategy_gets_zero_paired_accuracy():
    # a strategy that cannot see connectors predicts the same thing on orig and cf
    Q = query_table(_rows("C1", "C1"))
    m = metric_fns()
    assert m["acc_orig"](Q) == 1.0
    assert m["paired_cf_accuracy"](Q) == 0.0
    assert m["cf_stuck_rate"](Q) == 1.0
    assert m["invariance_style"](Q) == 1.0


def test_perfect_strategy():
    Q = query_table(_rows("C1", "C2"))
    m = metric_fns()
    assert m["paired_cf_accuracy"](Q) == 1.0
    assert m["cf_flip_given_orig_correct"](Q) == 1.0
    assert m["cf_stuck_rate"](Q) == 0.0


def test_compare_and_mcnemar():
    qa = query_table(_rows("C1", "C2", n=10))
    qb = query_table(_rows("C1", "C1", n=10))
    r = compare(qa, qb, "paired_cf_accuracy", B=200)
    assert r["diff_A_minus_B_pp"] == 100.0 and r["mcnemar_b"] == 10 and r["mcnemar_c"] == 0
    assert r["mcnemar_p"] < 0.01
    assert mcnemar_exact(0, 0) == 1.0


def test_normalize():
    assert normalize(' "4\'-0"" ') == normalize("4'-0\"")
    assert normalize("4′-0″") == normalize("4'-0\"")
    assert normalize("4'-0\"") != normalize("4'-10\"")
    assert is_correct("4.7 kΩ", "4.7kΩ")
    assert is_correct("c1", "C1")
    assert not is_correct("C1", "C12")


EX = {"query_type": "a2e", "candidate_entities": ["C1", "C2", "C12"], "candidate_values": ["4'-0\""]}


def test_parse_json_and_snap():
    assert parse('blah\n{"answer": "C2"}', EX, 100, 100)["prediction"] == "C2"
    assert parse("The annotation points to column C12.", EX, 100, 100)["prediction"] == "C12"
    assert parse("Answer: C1", EX, 100, 100)["prediction"] == "C1"
    # ambiguous mention is not snapped
    assert parse("either C1 or C2", EX, 100, 100)["prediction"] == "either C1 or C2"
    ex2 = {"query_type": "e2a", "candidate_entities": [], "candidate_values": ["4'-0\"", "12'-3\""]}
    assert parse('{"answer": "4\'-0""}', ex2, 100, 100)["prediction"] == "4'-0\""
    assert parse('{"answer": "NONE"}', ex2, 100, 100)["prediction"] == "NONE"


def test_parse_boxes_to_canvas():
    out = parse('{"answer": "C1", "entity_box": [0, 0, 500, 1000]}', EX, 1024, 720)
    assert out["pred_entity_bbox"] == [0.0, 0.0, 512.0, 720.0]
    out = parse('{"answer": "C1", "entity_box": [0, 0, 1000, 1000]}', EX, 400, 400, crop_box=[100, 100, 300, 200])
    assert out["pred_entity_bbox"] == [100.0, 100.0, 300.0, 200.0]
