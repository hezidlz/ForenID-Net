from __future__ import annotations

import argparse
import binascii
import csv
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_MASK_DIR = "generated_masks"


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def write_luma_png(path: Path, pixels: bytes, width: int, height: int) -> None:
    """Write an 8-bit grayscale PNG using only the Python standard library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        start = y * width
        rows.extend(pixels[start : start + width])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(bytes(rows), level=6))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def read_csv_records(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def bool_from_csv(value: str) -> bool:
    return value.strip().lower() == "true"


def normalize_rel(path: str) -> str:
    return Path(path.replace("\\", "/")).as_posix()


def jpeg_size(path: Path) -> tuple[int, int]:
    """Return JPEG dimensions without Pillow.

    FantasyID images are JPEGs. Keeping this stdlib fallback makes data
    preparation possible before the training environment has PIL installed.
    """
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"Not a JPEG file: {path}")
        while True:
            marker_start = f.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            length_bytes = f.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if marker in {
                b"\xc0",
                b"\xc1",
                b"\xc2",
                b"\xc3",
                b"\xc5",
                b"\xc6",
                b"\xc7",
                b"\xc9",
                b"\xca",
                b"\xcb",
                b"\xcd",
                b"\xce",
                b"\xcf",
            }:
                payload = f.read(5)
                if len(payload) != 5:
                    break
                height, width = struct.unpack(">HH", payload[1:5])
                return int(width), int(height)
            f.seek(length - 2, 1)
    raise ValueError(f"Could not read JPEG dimensions: {path}")


def image_size(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return jpeg_size(path)
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Pillow is required to read non-JPEG image size: {path}") from exc


def crop_size_from_json_obj(data: dict[str, Any]) -> tuple[int, int] | None:
    for key in ("cropping_info", "cropping_info-altered-recaptured"):
        info = data.get(key)
        if not isinstance(info, dict):
            continue
        width = info.get("resulted_cropped_image_width")
        height = info.get("resulted_cropped_image_height")
        if width and height:
            return int(width), int(height)
    return None


def crop_size_from_text(text: str) -> tuple[int, int] | None:
    width_match = re.search(r'"resulted_cropped_image_width"\s*:\s*(\d+)', text)
    height_match = re.search(r'"resulted_cropped_image_height"\s*:\s*(\d+)', text)
    if width_match and height_match:
        return int(width_match.group(1)), int(height_match.group(1))
    return None


def is_selected_field(field_name: str, mask_mode: str) -> bool:
    field = field_name.strip().lower()
    is_face = field == "face"
    if mask_mode == "all":
        return True
    if mask_mode == "face":
        return is_face
    if mask_mode == "text":
        return not is_face
    raise ValueError(f"Unknown mask mode: {mask_mode}")


def rect_from_region(region: dict[str, Any], mask_mode: str) -> tuple[int, int, int, int] | None:
    attrs = region.get("region_attributes") or {}
    if str(attrs.get("region_provenance", "")).lower() != "altered":
        return None
    if not is_selected_field(str(attrs.get("field_name", "")), mask_mode):
        return None

    shape = region.get("shape_attributes") or {}
    if shape.get("name", "rect") != "rect":
        return None
    try:
        return (
            int(round(float(shape["x"]))),
            int(round(float(shape["y"]))),
            int(round(float(shape["width"]))),
            int(round(float(shape["height"]))),
        )
    except (KeyError, TypeError, ValueError):
        return None


def altered_rects_from_json(data: dict[str, Any], mask_mode: str) -> list[tuple[int, int, int, int]]:
    regions = data.get("regions") or []
    rects = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        rect = rect_from_region(region, mask_mode)
        if rect is not None:
            rects.append(rect)
    return rects


def altered_rects_from_text(text: str, mask_mode: str) -> list[tuple[int, int, int, int]]:
    """Extract altered rectangles from malformed FantasyID JSON sidecars.

    Some sidecars contain broken text values, but the shape/provenance lines are
    still regular. This parser intentionally reads only the fields needed for
    mask generation.
    """
    number = r"[-+]?\d+(?:\.\d+)?"
    value_patterns = {
        "x": re.compile(rf'"x"\s*:\s*({number})'),
        "y": re.compile(rf'"y"\s*:\s*({number})'),
        "width": re.compile(rf'"width"\s*:\s*({number})'),
        "height": re.compile(rf'"height"\s*:\s*({number})'),
    }
    field_pattern = re.compile(r'"field_name"\s*:\s*"([^"]*)"')
    provenance_pattern = re.compile(r'"region_provenance"\s*:\s*"([^"]*)"')

    rects: list[tuple[int, int, int, int]] = []
    current: dict[str, Any] = {}
    for line in text.splitlines():
        for key, pattern in value_patterns.items():
            match = pattern.search(line)
            if match:
                current[key] = float(match.group(1))

        field_match = field_pattern.search(line)
        if field_match:
            current["field_name"] = field_match.group(1)

        provenance_match = provenance_pattern.search(line)
        if not provenance_match:
            continue

        provenance = provenance_match.group(1).lower()
        field_name = str(current.get("field_name", ""))
        has_rect = all(key in current for key in ("x", "y", "width", "height"))
        if provenance == "altered" and has_rect and is_selected_field(field_name, mask_mode):
            rects.append(
                (
                    int(round(current["x"])),
                    int(round(current["y"])),
                    int(round(current["width"])),
                    int(round(current["height"])),
                )
            )
        current = {}
    return rects


def load_altered_rects(json_path: Path, mask_mode: str) -> tuple[list[tuple[int, int, int, int]], tuple[int, int] | None, bool]:
    text = json_path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return altered_rects_from_json(data, mask_mode), crop_size_from_json_obj(data), False
    except json.JSONDecodeError:
        pass
    return altered_rects_from_text(text, mask_mode), crop_size_from_text(text), True


def mask_rel_path(mask_dir: str, image_rel: str) -> str:
    return f"{mask_dir}/{Path(image_rel).with_suffix('.png').as_posix()}"


def rects_to_mask(width: int, height: int, rects: list[tuple[int, int, int, int]]) -> bytes:
    mask = bytearray(width * height)
    for x, y, w, h in rects:
        x0 = max(0, min(width, x))
        y0 = max(0, min(height, y))
        x1 = max(0, min(width, x + max(0, w)))
        y1 = max(0, min(height, y + max(0, h)))
        if x1 <= x0 or y1 <= y0:
            continue
        row = b"\xff" * (x1 - x0)
        for yy in range(y0, y1):
            start = yy * width + x0
            mask[start : start + len(row)] = row
    return bytes(mask)


def prepare_split(
    root: Path,
    csv_name: str,
    out_path: Path,
    mask_dir: str,
    mask_mode: str,
    skip_existing: bool,
    no_masks: bool,
    limit: int,
) -> dict[str, int]:
    csv_path = root / csv_name
    records = read_csv_records(csv_path)
    if limit > 0:
        records = records[:limit]

    stats = {
        "rows": 0,
        "fake": 0,
        "real": 0,
        "masks_written": 0,
        "masks_skipped": 0,
        "fallback_json": 0,
        "empty_attack_masks": 0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for row in records:
            image_rel = normalize_rel(row["path"])
            image_path = root / image_rel
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing FantasyID image: {image_path}")

            is_attack = bool_from_csv(row["is_attack"])
            label = 1 if is_attack else 0
            stats["rows"] += 1
            stats["fake" if is_attack else "real"] += 1

            if not is_attack or no_masks:
                out.write(f"{image_rel},None,{label}\n")
                continue

            json_path = image_path.with_suffix(".json")
            if not json_path.is_file():
                raise FileNotFoundError(f"Missing FantasyID JSON sidecar: {json_path}")

            rects, crop_size, used_fallback = load_altered_rects(json_path, mask_mode)
            if used_fallback:
                stats["fallback_json"] += 1
            if not rects:
                stats["empty_attack_masks"] += 1

            width, height = crop_size if crop_size is not None else image_size(image_path)
            rel_mask = mask_rel_path(mask_dir, image_rel)
            mask_path = root / rel_mask
            if skip_existing and mask_path.is_file():
                stats["masks_skipped"] += 1
            else:
                mask = rects_to_mask(width=width, height=height, rects=rects)
                write_luma_png(mask_path, mask, width=width, height=height)
                stats["masks_written"] += 1
            out.write(f"{image_rel},{rel_mask},{label}\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser("Prepare FantasyID finetuning lists and box masks for ForenID-Net")
    parser.add_argument("--root", default="datasets/FantasyID/FantasyID")
    parser.add_argument("--out-dir", default="data_lists")
    parser.add_argument("--train-csv", default="train.csv")
    parser.add_argument("--valid-csv", default="test.csv", help="This local copy has no val.csv, so test.csv is used as validation by default.")
    parser.add_argument("--train-out", default="fantasyid_train.txt")
    parser.add_argument("--valid-out", default="fantasyid_valid.txt")
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--mask-mode", choices=("all", "face", "text"), default="all")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing FantasyID mask PNGs.")
    parser.add_argument("--no-masks", action="store_true", help="Write image-level labels only.")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per CSV; 0 means no limit.")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"FantasyID root does not exist: {root}")

    train_stats = prepare_split(
        root=root,
        csv_name=args.train_csv,
        out_path=out_dir / args.train_out,
        mask_dir=args.mask_dir,
        mask_mode=args.mask_mode,
        skip_existing=not args.overwrite,
        no_masks=args.no_masks,
        limit=args.limit,
    )
    valid_stats = prepare_split(
        root=root,
        csv_name=args.valid_csv,
        out_path=out_dir / args.valid_out,
        mask_dir=args.mask_dir,
        mask_mode=args.mask_mode,
        skip_existing=not args.overwrite,
        no_masks=args.no_masks,
        limit=args.limit,
    )

    print(f"root: {root}")
    print(f"train list: {out_dir / args.train_out} {train_stats}")
    print(f"valid list: {out_dir / args.valid_out} {valid_stats}")
    if args.valid_csv == "test.csv":
        print("note: local FantasyID has train.csv/test.csv only; test.csv was written as the validation list.")


if __name__ == "__main__":
    main()
