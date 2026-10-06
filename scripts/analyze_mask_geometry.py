from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def largest_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit box-derived mask geometry by protocol split.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for manifest_path in sorted(args.manifest_dir.glob("fantasyid_*.csv")):
        frame = pd.read_csv(manifest_path)
        for row in frame.itertuples(index=False):
            if int(row.label) != 1 or not isinstance(row.mask, str) or not row.mask:
                continue
            mask_path = args.root / row.mask
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L")) > 127
            height, width = mask.shape
            box = largest_box(mask)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            records.append(
                {
                    "split": row.split,
                    "path": row.path,
                    "group_key": row.group_key,
                    "attack_type": row.attack_type,
                    "device": row.device,
                    "height": height,
                    "width": width,
                    "positive_pixels": int(mask.sum()),
                    "area_ratio": float(mask.mean()),
                    "enclosing_box_width": x1 - x0,
                    "enclosing_box_height": y1 - y0,
                    "enclosing_box_area_ratio": float((x1 - x0) * (y1 - y0) / (height * width)),
                }
            )

    detail = pd.DataFrame(records)
    args.output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output / "mask_geometry_per_image.csv", index=False)
    summaries: list[dict[str, object]] = []
    for split, frame in detail.groupby("split", sort=True):
        ratios = frame["area_ratio"].to_numpy()
        summaries.append(
            {
                "split": split,
                "fake_images_with_mask": len(frame),
                "groups": frame["group_key"].nunique(),
                "area_ratio_min": float(np.min(ratios)),
                "area_ratio_q05": float(np.quantile(ratios, 0.05)),
                "area_ratio_q25": float(np.quantile(ratios, 0.25)),
                "area_ratio_median": float(np.median(ratios)),
                "area_ratio_q75": float(np.quantile(ratios, 0.75)),
                "area_ratio_q95": float(np.quantile(ratios, 0.95)),
                "area_ratio_max": float(np.max(ratios)),
                "n_le_0_5pct": int(np.sum(ratios <= 0.005)),
                "n_le_1pct": int(np.sum(ratios <= 0.01)),
                "n_le_2pct": int(np.sum(ratios <= 0.02)),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "mask_geometry_summary.csv", index=False)
    (args.output / "mask_geometry_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
