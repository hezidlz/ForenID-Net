from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_records(path: Path) -> list[tuple[str, str | None, int]]:
    records: list[tuple[str, str | None, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                raise ValueError(f"Invalid record at {path}:{line_no}: {raw_line!r}")
            image = parts[0]
            mask = None if parts[1].lower() in {"none", "negative", "null", ""} else parts[1]
            label = int(float(parts[2])) if len(parts) > 2 and parts[2] != "" else (0 if mask is None else 1)
            records.append((image, mask, label))
    return records


def prefix_records(records: list[tuple[str, str | None, int]], prefix: str) -> list[tuple[str, str | None, int]]:
    prefix = prefix.strip("/\\")
    out: list[tuple[str, str | None, int]] = []
    for image, mask, label in records:
        image_path = f"{prefix}/{image}".replace("\\", "/")
        mask_path = f"{prefix}/{mask}".replace("\\", "/") if mask is not None else None
        out.append((image_path, mask_path, label))
    return out


def validate_records(root: Path, records: list[tuple[str, str | None, int]], require_masks: bool = True) -> None:
    missing: list[Path] = []
    for image, mask, label in records:
        image_path = root / image
        if not image_path.is_file():
            missing.append(image_path)
        if mask is not None:
            mask_path = root / mask
            if not mask_path.is_file():
                missing.append(mask_path)
        elif label == 1 and require_masks:
            missing.append(root / f"<missing-mask-for>/{image}")
    if missing:
        preview = "\n".join(str(path) for path in missing[:30])
        raise FileNotFoundError(f"Missing {len(missing)} files or masks:\n{preview}")


def write_records(path: Path, records: list[tuple[str, str | None, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for image, mask, label in records:
            mask_text = mask if mask is not None else "None"
            f.write(f"{image},{mask_text},{label}\n")


def summarize(name: str, records: list[tuple[str, str | None, int]]) -> str:
    fake = sum(1 for _, _, label in records if label == 1)
    real = len(records) - fake
    masked_fake = sum(1 for _, mask, label in records if label == 1 and mask is not None)
    return f"{name}: total={len(records)} fake={fake} real={real} masked_fake={masked_fake}"


def maybe_limit(records: list[tuple[str, str | None, int]], limit: int) -> list[tuple[str, str | None, int]]:
    if limit <= 0:
        return records
    fake = [record for record in records if record[2] == 1][:limit]
    real = [record for record in records if record[2] == 0][:limit]
    return fake + real


def main() -> None:
    parser = argparse.ArgumentParser("Build a combined IMD2020/CASIA/FantasticReality pretrain list")
    parser.add_argument("--root", default=".", help="Root used by the combined txt list.")
    parser.add_argument("--out-dir", default="data_lists")
    parser.add_argument("--train-out", default="general_pretrain_train.txt")
    parser.add_argument("--valid-out", default="general_pretrain_valid.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-dataset", type=int, default=0, help="Debug limit per fake/real split.")
    parser.add_argument("--allow-missing-fake-masks", action="store_true")
    parser.add_argument("--imd-train", default="data_lists/imd2020_train.txt")
    parser.add_argument("--imd-valid", default="data_lists/imd2020_valid.txt")
    parser.add_argument("--casia-train", default="data_lists/casia_v2_train.txt")
    parser.add_argument("--casia-valid", default="data_lists/casia_v2_valid.txt")
    parser.add_argument("--fr-train", default="data_lists/fr_train.txt")
    parser.add_argument("--fr-valid", default="data_lists/fr_valid.txt")
    args = parser.parse_args()

    sources = [
        ("imd2020", Path(args.imd_train), Path(args.imd_valid), "datasets/IMD2020"),
        ("casia_v2", Path(args.casia_train), Path(args.casia_valid), "datasets/CASIA/CASIA"),
        ("fantasticreality", Path(args.fr_train), Path(args.fr_valid), "datasets/FantasticReality/FantasticReality_v1/dataset"),
    ]

    train_records: list[tuple[str, str | None, int]] = []
    valid_records: list[tuple[str, str | None, int]] = []
    for name, train_list, valid_list, prefix in sources:
        train = maybe_limit(prefix_records(read_records(train_list), prefix), args.limit_per_dataset)
        valid = maybe_limit(prefix_records(read_records(valid_list), prefix), args.limit_per_dataset)
        print(summarize(f"{name} train", train))
        print(summarize(f"{name} valid", valid))
        train_records.extend(train)
        valid_records.extend(valid)

    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    rng.shuffle(valid_records)

    require_masks = not args.allow_missing_fake_masks
    validate_records(Path(args.root), train_records + valid_records, require_masks=require_masks)

    out_dir = Path(args.out_dir)
    write_records(out_dir / args.train_out, train_records)
    write_records(out_dir / args.valid_out, valid_records)
    print(summarize("combined train", train_records))
    print(summarize("combined valid", valid_records))
    print(f"train list: {out_dir / args.train_out}")
    print(f"valid list: {out_dir / args.valid_out}")


if __name__ == "__main__":
    main()
