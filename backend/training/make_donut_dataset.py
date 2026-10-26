"""Render the synthetic documents as noisy "phone photo" images for Donut fine-tuning.

    python -m training.make_donut_dataset --out data/donut --n 300

Writes <out>/{train,validation}/<id>.png and metadata.jsonl in the Hugging Face
imagefolder format: {"file_name": ..., "ground_truth": "{\"gt_parse\": {...}}"}.
Mix in real labelled scans before trusting any model trained on this.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from evals.dataset import build_shipments

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def _font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render(text: str, rng: random.Random) -> Image.Image:
    font = _font(rng.randint(16, 22))
    lines = text.splitlines()
    width = 40 + max(int(font.getlength(line)) for line in lines if line) + 40
    height = 40 + len(lines) * int(font.size * 1.5) + 40
    paper = rng.randint(225, 250)
    img = Image.new("L", (width, height), paper)
    draw = ImageDraw.Draw(img)
    y = 40
    for line in lines:
        draw.text((40 + rng.randint(-2, 2), y), line, fill=rng.randint(10, 60), font=font)
        y += int(font.size * 1.5)
    # Phone-camera artefacts: slight rotation, blur, uneven lighting.
    img = img.rotate(rng.uniform(-2.5, 2.5), expand=True, fillcolor=paper)
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    gradient = Image.linear_gradient("L").resize(img.size)
    img = Image.blend(img, ImageChops.multiply(img, gradient), rng.uniform(0.05, 0.25))
    return img.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/donut")
    ap.add_argument("--n", type=int, default=300, help="number of shipments (3 documents each)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    counts = {"train": 0, "validation": 0}
    for split in counts:
        (out / split).mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[str]] = {split: [] for split in counts}
    for ship in build_shipments(args.n, seed=args.seed):
        for doc in ship.docs:
            split = "validation" if rng.random() < 0.1 else "train"
            name = f"{doc.id}.png"
            render(doc.text, rng).save(out / split / name)
            gt = {"gt_parse": {"doc_type": doc.doc_type, **{k: v for k, v in doc.expected_fields.items() if v is not None}}}
            rows[split].append(json.dumps({"file_name": name, "ground_truth": json.dumps(gt)}))
            counts[split] += 1
    for split, lines in rows.items():
        (out / split / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
