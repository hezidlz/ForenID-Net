from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import imagehash
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_key(row: dict[str, str]) -> str:
    return Path(row["path"]).stem


def template_key(group: str) -> str:
    return group.split("-", 1)[0].lower()


def bool_label(row: dict[str, str]) -> int:
    return int(row["is_attack"].strip().lower() in {"1", "true", "yes"})


def device(row: dict[str, str]) -> str:
    return Path(row["path"]).parts[-2]


def mask_path(row: dict[str, str]) -> str | None:
    if not bool_label(row):
        return None
    return (Path("generated_masks") / Path(row["path"])).with_suffix(".png").as_posix()


def split_groups(groups: list[str], seed: int) -> tuple[set[str], set[str], set[str]]:
    strata = [template_key(item) for item in groups]
    first = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, held_idx = next(first.split(groups, strata))
    train = {groups[index] for index in train_idx}
    held = [groups[index] for index in held_idx]
    held_strata = [template_key(item) for item in held]
    second = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    dev_idx, test_idx = next(second.split(held, held_strata))
    dev = {held[index] for index in dev_idx}
    test = {held[index] for index in test_idx}
    assert not (train & dev or train & test or dev & test)
    assert train | dev | test == set(groups)
    return train, dev, test


def write_list(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            mask = mask_path(row) or "None"
            handle.write(f'{row["path"]},{mask},{bool_label(row)}\n')


def write_manifest(path: Path, rows: Iterable[dict[str, str]], split: str, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "source_csv",
        "path",
        "mask",
        "label",
        "attack_type",
        "device",
        "group_key",
        "template_stratum",
        "mask_provenance",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": split,
                    "source_csv": source,
                    "path": row["path"],
                    "mask": mask_path(row) or "",
                    "label": bool_label(row),
                    "attack_type": row.get("attack_type", ""),
                    "device": device(row),
                    "group_key": group_key(row),
                    "template_stratum": template_key(group_key(row)),
                    "mask_provenance": "box-derived" if bool_label(row) else "none",
                }
            )


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "samples": len(rows),
        "groups": len({group_key(row) for row in rows}),
        "labels": dict(sorted(Counter(str(bool_label(row)) for row in rows).items())),
        "attack_types": dict(sorted(Counter(row.get("attack_type", "") for row in rows).items())),
        "devices": dict(sorted(Counter(device(row) for row in rows).items())),
        "templates": dict(sorted(Counter(template_key(group_key(row)) for row in rows).items())),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_audit(root: Path, manifests: dict[str, list[dict[str, str]]], out_dir: Path, phash_distance: int) -> dict[str, object]:
    records: list[dict[str, str]] = []
    for split, rows in manifests.items():
        for row in rows:
            image_path = root / row["path"]
            with Image.open(image_path) as image:
                perceptual = str(imagehash.phash(image.convert("RGB")))
            records.append(
                {
                    "split": split,
                    "path": row["path"],
                    "group": group_key(row),
                    "sha256": sha256(image_path),
                    "phash": perceptual,
                }
            )

    exact: list[dict[str, str]] = []
    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_sha[record["sha256"]].append(record)
    for digest, items in by_sha.items():
        if len({item["split"] for item in items}) > 1:
            for item in items:
                exact.append({**item, "duplicate_sha256": digest})

    near: list[dict[str, object]] = []
    parsed = [(item, imagehash.hex_to_hash(item["phash"])) for item in records]
    for index, (left, left_hash) in enumerate(parsed):
        for right, right_hash in parsed[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            distance = left_hash - right_hash
            if distance <= phash_distance:
                near.append(
                    {
                        "left_split": left["split"],
                        "left_path": left["path"],
                        "left_group": left["group"],
                        "right_split": right["split"],
                        "right_path": right["path"],
                        "right_group": right["group"],
                        "phash_distance": distance,
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("exact_duplicates.csv", exact), ("near_duplicates.csv", near)):
        path = out_dir / name
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            if fieldnames:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    return {"exact_cross_split_records": len(exact), "near_cross_split_pairs": len(near), "phash_distance": phash_distance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--skip-hash-audit", action="store_true")
    parser.add_argument("--phash-distance", type=int, default=4)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    official_train = read_csv(root / "train.csv")
    official_test = read_csv(root / "test.csv")
    train_groups_all = sorted({group_key(row) for row in official_train})
    train_groups, dev_groups, internal_test_groups = split_groups(train_groups_all, args.seed)

    by_test_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in official_test:
        by_test_group[group_key(row)].append(row)

    train_rows = [row for row in official_train if group_key(row) in train_groups]
    dev_rows = [row for row in official_train if group_key(row) in dev_groups]
    internal_test_rows = [row for row in official_train if group_key(row) in internal_test_groups]
    for group in sorted(dev_groups):
        dev_rows.extend(by_test_group.get(group, []))
    for group in sorted(internal_test_groups):
        internal_test_rows.extend(by_test_group.get(group, []))

    external_groups = {
        group
        for group, rows in by_test_group.items()
        if group not in set(train_groups_all)
        and any(not bool_label(row) for row in rows)
        and any(bool_label(row) for row in rows)
    }
    external_rows = [row for row in official_test if group_key(row) in external_groups]
    conditional_only_rows = [
        row
        for row in official_test
        if group_key(row) not in external_groups
        and group_key(row) not in dev_groups
        and group_key(row) not in internal_test_groups
    ]

    partitions = {
        "train": train_rows,
        "dev": dev_rows,
        "internal_test": internal_test_rows,
        "external_clean_test": external_rows,
    }
    list_dir = output / "lists"
    manifest_dir = output / "manifests"
    for split, rows in partitions.items():
        write_list(list_dir / f"fantasyid_{split}.txt", rows)
        source = "train.csv+test.csv" if split in {"dev", "internal_test"} else ("train.csv" if split == "train" else "test.csv")
        write_manifest(manifest_dir / f"fantasyid_{split}.csv", rows, split, source)
    write_list(list_dir / "fantasyid_conditional_localization_only.txt", conditional_only_rows)
    write_manifest(
        manifest_dir / "fantasyid_conditional_localization_only.csv",
        conditional_only_rows,
        "conditional_localization_only",
        "test.csv",
    )

    group_sets = {name: {group_key(row) for row in rows} for name, rows in partitions.items()}
    overlap_matrix: dict[str, int] = {}
    names = list(group_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap_matrix[f"{left}__{right}"] = len(group_sets[left] & group_sets[right])
    if any(overlap_matrix.values()):
        raise RuntimeError(f"Group leakage detected: {overlap_matrix}")

    summary: dict[str, object] = {
        "seed": args.seed,
        "group_definition": "filename stem",
        "template_definition": "prefix before first hyphen",
        "partitions": {name: summarize(rows) for name, rows in partitions.items()},
        "conditional_localization_only": summarize(conditional_only_rows),
        "group_overlap_matrix": overlap_matrix,
    }
    if not args.skip_hash_audit:
        summary["hash_audit"] = hash_audit(root, partitions, output / "leakage_audit", args.phash_distance)
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
