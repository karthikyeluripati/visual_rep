# Real-diagram labelling guide (RelBind external-validity set)

You will mark where reference-label leader lines end in technical drawings. You do **not** judge any model output.
About 1–2 minutes per figure. The target is **100 figures**, two labellers, 20 figures labelled by both to measure agreement.

## 1. Choose figures

- **Source:** US patent drawings (public domain). Use Google Patents, choose a patent, open "Drawings", and download the figure PNG.
- **Keep** a figure if it has **at least 4 numeric reference labels** (e.g. 12, 14, 16), each with a visible **leader line** to a part.
- **Skip** figures where:
  - two labels carry the same number;
  - labels have no lines, or the lines are only arrows into empty space;
  - the figure is a photo.
- **Prefer** variety: mechanical, electrical, chemical apparatus, flow diagrams; dense and sparse; crossing leader lines.
- Save the image as `data/real/img/<patentnumber>_fig<k>.png`. Record its URL.

## 2. Label each figure (one JSON file per figure, in `data/real/labels/`)

```json
{
  "image": "img/US1234567_fig2.png",
  "source": "https://patents.google.com/patent/US1234567/en",
  "license": "public domain (US patent)",
  "labeller": "L1",
  "callouts": [
    {"text": "12", "text_bbox": [x0, y0, x1, y1], "end": [x, y]},
    {"text": "14", "text_bbox": [x0, y0, x1, y1], "end": [x, y]}
  ]
}
```

Fields:
- `text`: the label exactly as printed.
- `text_bbox`: a tight box around the label's digits only, in **pixel coordinates** of the saved PNG (x0, y0 top-left; x1, y1 bottom-right). Leave a 1–2 px margin and do **not** include the leader line.
- `end`: the pixel where the leader line touches the part it designates, which is the line's far end from the label.

Rules:
- Label **every** numeric callout that has a leader line, not just 4.
- Any box tool that exports pixel coordinates works, for example Label Studio (rectangle and keypoint), makesense.ai, or GIMP pointer coordinates.
- If you are unsure where a line ends (for example, it merges into hatching), **leave that callout out** and note it in `"notes"`.

## 3. What happens next (automatic)

`python -m relbind.real build` turns each figure into test items:
- It draws four lettered markers (A–D) at line endpoints.
- It asks "which marker is at the end of the leader line from label 12?"
- It makes a counterfactual image by swapping the pixels of two labels' digits. Lines stay identical, and the answer flips.

The 20 doubly-labelled figures are used to report labeller agreement on `end` points (pixel distance) before any model is scored.
