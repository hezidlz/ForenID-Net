from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_imd2020_samples(root: Path) -> dict[str, list[tuple[str, str | None, int]]]:
    if not root.is_dir():
        raise ValueError(f"IMD2020 root does not exist: {root}")

    samples_by_case: dict[str, list[tuple[str, str | None, int]]] = {}
    case_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for case_dir in case_dirs:
        case_samples: list[tuple[str, str | None, int]] = []
        files = [p for p in case_dir.iterdir() if p.is_file()]
        image_files = [
            p
            for p in files
            if p.suffix.lower() in IMAGE_EXTS and not p.stem.endswith("_mask")
        ]

        for image_path in sorted(image_files):
            rel_image = image_path.relative_to(root).as_posix()
            if "_orig" in image_path.stem:
                case_samples.append((rel_image, None, 0))
                continue

            mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
            if not mask_path.is_file():
                print(f"[skip] missing mask for {image_path}")
                continue
            rel_mask = mask_path.relative_to(root).as_posix()
            case_samples.append((rel_image, rel_mask, 1))

        if case_samples:
            samples_by_case[case_dir.name] = case_samples
    return samples_by_case


def split_cases(case_names: list[str], valid_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(case_names)
    rng.shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_ratio)))
    valid_cases = sorted(shuffled[:valid_count])
    train_cases = sorted(shuffled[valid_count:])
    return train_cases, valid_cases


def write_list(path: Path, samples: list[tuple[str, str | None, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for image, mask, label in samples:
            mask_text = mask if mask is not None else "None"
            f.write(f"{image},{mask_text},{label}\n")


def main() -> None:
    parser = argparse.ArgumentParser("Prepare IMD2020 txt lists for ForenID-Net")
    parser.add_argument("--root", default="datasets/IMD2020", help="IMD2020 dataset root")
    parser.add_argument("--out-dir", default="data_lists", help="Directory for generated txt lists")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    samples_by_case = find_imd2020_samples(root)
    train_cases, valid_cases = split_cases(sorted(samples_by_case), args.valid_ratio, args.seed)

    train_samples = [sample for case in train_cases for sample in samples_by_case[case]]
    valid_samples = [sample for case in valid_cases for sample in samples_by_case[case]]

    write_list(out_dir / "imd2020_train.txt", train_samples)
    write_list(out_dir / "imd2020_valid.txt", valid_samples)

    def count(samples: list[tuple[str, str | None, int]]) -> tuple[int, int]:
        real = sum(1 for _, _, label in samples if label == 0)
        fake = sum(1 for _, _, label in samples if label == 1)
        return real, fake

    train_real, train_fake = count(train_samples)
    valid_real, valid_fake = count(valid_samples)
    print(f"IMD2020 root: {root}")
    print(f"Cases: {len(samples_by_case)} -> train {len(train_cases)}, valid {len(valid_cases)}")
    print(f"Train: {len(train_samples)} samples, real={train_real}, fake={train_fake}")
    print(f"Valid: {len(valid_samples)} samples, real={valid_real}, fake={valid_fake}")
    print(f"Wrote: {out_dir / 'imd2020_train.txt'}")
    print(f"Wrote: {out_dir / 'imd2020_valid.txt'}")


if __name__ == "__main__":
    main()
