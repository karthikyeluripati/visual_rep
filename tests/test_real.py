"""Real-diagram item builder: strict-cf properties on a synthetic fixture (NOT a real diagram)."""
import json

import numpy as np
from PIL import Image, ImageDraw

from relbind import real


def _fixture(tmp_path):
    im = Image.new("RGB", (600, 400), "white")
    d = ImageDraw.Draw(im)
    cs = []
    for k, (tx, ty, ex, ey) in enumerate([(40, 40, 300, 200), (40, 120, 320, 300), (520, 40, 250, 120), (520, 320, 400, 250), (40, 330, 150, 220)]):
        t = str(10 + 2 * k)
        d.text((tx, ty), t, fill="black")
        bb = d.textbbox((tx, ty), t)
        bb = [bb[0] - 2, bb[1] - 2, bb[2] + 2, bb[3] + 2]
        d.line([(bb[2] if tx < ex else bb[0], (bb[1] + bb[3]) / 2), (ex, ey)], fill="black", width=2)
        cs.append({"text": t, "text_bbox": bb, "end": [ex, ey]})
    (tmp_path / "img").mkdir()
    im.save(tmp_path / "img" / "fx.png")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "fx.json").write_text(json.dumps({"image": "img/fx.png", "source": "synthetic test fixture",
                                                             "license": "n/a", "callouts": cs}))
    return cs


def test_build_strict_cf(tmp_path):
    cs = _fixture(tmp_path)
    exs = real.build(tmp_path / "labels", tmp_path, tmp_path / "items", seed=0)
    assert exs and len(exs) % 4 == 0
    by_q = {}
    for e in exs:
        by_q.setdefault(e["query_id"], {})[e["variant"]] = e
    for q in by_q.values():
        assert q["orig"]["answer"] != q["cf"]["answer"]  # flips
        assert q["orig"]["candidate_entities"] == q["cf"]["candidate_entities"]
    # cf differs from orig only inside the two swapped text boxes
    g = exs[0]["group_id"]
    a = np.asarray(Image.open(tmp_path / "items" / "images" / f"{g}-orig.png")).astype(int)
    b = np.asarray(Image.open(tmp_path / "items" / "images" / f"{g}-cf.png")).astype(int)
    texts = {e["query_text"] for e in exs if e["group_id"] == g}
    mask = np.zeros(a.shape[:2], bool)
    for c in cs:
        if c["text"] in texts:
            x0, y0, x1, y1 = (int(round(v)) for v in c["text_bbox"])
            mask[y0:y1, x0:x1] = True
    diff = np.abs(a - b).sum(-1) > 0
    assert diff.any() and not diff[~mask].any()
    # deterministic
    exs2 = real.build(tmp_path / "labels", tmp_path, tmp_path / "items2", seed=0)
    assert [e["answer"] for e in exs] == [e["answer"] for e in exs2]
