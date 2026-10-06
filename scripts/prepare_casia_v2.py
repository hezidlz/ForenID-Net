from __future__ import annotations

import argparse
from pathlib import Path


def default_list_dir() -> Path:
    return Path("../TruFor-main/TruFor_train_test/dataset/data")


def read_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_record(line: str) -> tuple[str, str | None]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 'image,mask' record, got: {line!r}")
    image = parts[0]
    mask = None if parts[1].lower() in {"none", "negative", "null", ""} else parts[1]
    return image, mask


def localize_image_path(path: str) -> str:
    if path.startswith("Tp/"):
        return "CASIA 2.0/" + path
    if path.startswith("Au/"):
        return "CASIA 2.0/" + path
    raise ValueError(f"Unsupported CASIA image path: {path}")


def localize_mask_path(path: str | None) -> str | None:
    if path is None:
        return None
    if path.startswith("groundtruth/"):
        return path.replace("groundtruth/", "CASIA 2 Groundtruth/", 1)
    raise ValueError(f"Unsupported CASIA mask path: {path}")


def build_records(fake_list: Path, real_list: Path) -> list[tuple[str, str | None, int]]:
    records: list[tuple[str, str | None, int]] = []
    for line in read_list(fake_list):
        image, mask = parse_record(line)
        records.append((localize_image_path(image), localize_mask_path(mask), 1))
    for line in read_list(real_list):
        image, mask = parse_record(line)
        if mask is not None:
            raise ValueError(f"Authentic CASIA record should not have a mask: {line!r}")
        records.append((localize_image_path(image), None, 0))
    return records


def validate_records(root: Path, records: list[tuple[str, str | None, int]]) -> None:
    missing: list[Path] = []
    for image, mask, _ in records:
        image_path = root / image
        if not image_path.is_file():
            missing.append(image_path)
        if mask is not None:
            mask_path = root / mask
            if not mask_path.is_file():
                missing.append(mask_path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} CASIA files:\n{preview}")


def write_records(path: Path, records: list[tuple[str, str | None, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for image, mask, label in records:
            mask_text = mask if mask is not None else "None"
            f.write(f"{image},{mask_text},{label}\n")


def main() -> None:
    parser = argparse.ArgumentParser("Prepare CASIA v2 txt lists for ForenID-Net")
    parser.add_argument("--root", default="datasets/CASIA/CASIA")
    parser.add_argument("--list-dir", default=str(default_list_dir()))
    parser.add_argument("--out-dir", default="data_lists")
    parser.add_argument("--train-out", default="casia_v2_train.txt")
    parser.add_argument("--valid-out", default="casia_v2_valid.txt")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per fake/real split; 0 means no limit.")
    args = parser.parse_args()

    root = Path(args.root)
    list_dir = Path(args.list_dir)
    out_dir = Path(args.out_dir)

    train_records = build_records(
        list_dir / "CASIA_v2_train_list.txt",
        list_dir / "CASIA_v2_auth_train_list.txt",
    )
    valid_records = build_records(
        list_dir / "CASIA_v2_valid_list.txt",
        list_dir / "CASIA_v2_auth_valid_list.txt",
    )
    if args.limit > 0:
        train_fake = [record for record in train_records if record[2] == 1][: args.limit]
        train_real = [record for record in train_records if record[2] == 0][: args.limit]
        valid_fake = [record for record in valid_records if record[2] == 1][: args.limit]
        valid_real = [record for record in valid_records if record[2] == 0][: args.limit]
        train_records = train_fake + train_real
        valid_records = valid_fake + valid_real

    validate_records(root, train_records + valid_records)
    write_records(out_dir / args.train_out, train_records)
    write_records(out_dir / args.valid_out, valid_records)

    train_fake = sum(1 for _, _, label in train_records if label == 1)
    train_real = len(train_records) - train_fake
    valid_fake = sum(1 for _, _, label in valid_records if label == 1)
    valid_real = len(valid_records) - valid_fake
    print(f"root: {root}")
    print(f"train list: {out_dir / args.train_out} fake={train_fake} real={train_real}")
    print(f"valid list: {out_dir / args.valid_out} fake={valid_fake} real={valid_real}")


if __name__ == "__main__":
    main()
