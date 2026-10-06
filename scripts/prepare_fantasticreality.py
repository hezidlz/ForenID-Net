from __future__ import annotations

import argparse
import ast
import binascii
import io
import struct
import zipfile
import zlib
from pathlib import Path


IMAGE_EXT = ".jpg"
NPZ_EXT = ".npz"
PNG_EXT = ".png"


def read_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def default_list_dir() -> Path:
    return Path("../TruFor-main/TruFor_train_test/dataset/data")


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def write_luma_png(path: Path, pixels: bytes, width: int, height: int) -> None:
    """Write an 8-bit grayscale PNG using only the Python standard library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = bytearray()
    stride = width
    for y in range(height):
        rows.append(0)  # PNG filter type 0 for each row.
        start = y * stride
        rows.extend(pixels[start : start + stride])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(bytes(rows), level=6))
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)


def read_npy_from_npz(path: Path) -> tuple[dict, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".npy")]
        if not names:
            raise ValueError(f"No .npy payload found in {path}")
        raw = archive.read(names[0])

    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"Invalid npy payload in {path}")
    version = raw[6:8]
    if version == b"\x01\x00":
        header_len = struct.unpack("<H", raw[8:10])[0]
        offset = 10
    elif version in {b"\x02\x00", b"\x03\x00"}:
        header_len = struct.unpack("<I", raw[8:12])[0]
        offset = 12
    else:
        raise ValueError(f"Unsupported npy version {version!r} in {path}")

    header = ast.literal_eval(raw[offset : offset + header_len].decode("latin1"))
    payload = raw[offset + header_len :]
    return header, payload


def squeeze_mask_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    squeezed = tuple(dim for dim in shape if dim != 1)
    if len(squeezed) != 2:
        raise ValueError(f"Expected a 2D mask after squeeze, got shape={shape}")
    return int(squeezed[0]), int(squeezed[1])


def convert_npz_mask_stdlib(npz_path: Path, png_path: Path) -> None:
    """Convert FantasticReality uint8 npz masks to binary PNG without numpy/PIL."""
    header, payload = read_npy_from_npz(npz_path)
    if header.get("fortran_order"):
        raise ValueError(f"Fortran-order masks are not supported: {npz_path}")
    if header.get("descr") != "|u1":
        raise ValueError(f"Only uint8 masks are supported by stdlib converter: {npz_path}")

    height, width = squeeze_mask_shape(tuple(header["shape"]))
    expected = height * width
    if len(payload) != expected:
        raise ValueError(f"Mask payload size mismatch in {npz_path}: {len(payload)} != {expected}")

    mask = bytes(255 if value > 0 else 0 for value in payload)
    write_luma_png(png_path, mask, width=width, height=height)


def convert_npz_mask(npz_path: Path, png_path: Path) -> None:
    """Convert one npz mask to a binary PNG.

    Prefer numpy/PIL when installed, but keep a stdlib fallback because the
    repository often needs data preparation before the training environment is
    fully configured.
    """
    try:
        import numpy as np
        from PIL import Image

        payload = np.load(npz_path)
        key = "arr_0" if "arr_0" in payload else payload.files[0]
        mask = payload[key].squeeze()
        mask = (mask > 0).astype(np.uint8) * 255
        png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, mode="L").save(png_path)
    except ModuleNotFoundError:
        convert_npz_mask_stdlib(npz_path, png_path)


def fake_mask_png_name(fake_name: str) -> str:
    return Path(fake_name).with_suffix(PNG_EXT).name


def fake_mask_npz_name(fake_name: str) -> str:
    return Path(fake_name).with_suffix(NPZ_EXT).name


def validate_paths(root: Path, fake_names: list[str], real_names: list[str]) -> None:
    missing: list[Path] = []
    for name in fake_names:
        image_path = root / "ColorFakeImages" / name
        mask_path = root / "SegmentationFake" / fake_mask_npz_name(name)
        if not image_path.is_file():
            missing.append(image_path)
        if not mask_path.is_file():
            missing.append(mask_path)
    for name in real_names:
        image_path = root / "ColorRealImages" / name
        if not image_path.is_file():
            missing.append(image_path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} FantasticReality files:\n{preview}")


def write_doc_lite_list(path: Path, fake_names: list[str], real_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for name in fake_names:
            mask_name = fake_mask_png_name(name)
            f.write(f"ColorFakeImages/{name},SegmentationFakePng/{mask_name},1\n")
        for name in real_names:
            f.write(f"ColorRealImages/{name},None,0\n")


def convert_masks(root: Path, fake_names: list[str], skip_existing: bool = True) -> tuple[int, int]:
    converted = 0
    skipped = 0
    out_dir = root / "SegmentationFakePng"
    for idx, name in enumerate(fake_names, start=1):
        src = root / "SegmentationFake" / fake_mask_npz_name(name)
        dst = out_dir / fake_mask_png_name(name)
        if skip_existing and dst.is_file():
            skipped += 1
            continue
        convert_npz_mask(src, dst)
        converted += 1
        if converted % 250 == 0:
            print(f"converted {converted} masks, current={dst.name}")
    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser("Prepare FantasticReality for ForenID-Net")
    parser.add_argument("--root", default="datasets/FantasticReality/FantasticReality_v1/dataset")
    parser.add_argument("--list-dir", default=str(default_list_dir()))
    parser.add_argument("--out-dir", default="data_lists")
    parser.add_argument("--train-out", default="fr_train.txt")
    parser.add_argument("--valid-out", default="fr_valid.txt")
    parser.add_argument("--convert-all", action="store_true", help="Convert all SegmentationFake masks, not just listed masks.")
    parser.add_argument("--no-convert", action="store_true", help="Only write txt lists.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing PNG masks.")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per split/list; 0 means no limit.")
    args = parser.parse_args()

    root = Path(args.root)
    list_dir = Path(args.list_dir)
    out_dir = Path(args.out_dir)

    fake_train = read_list(list_dir / "FR_train_list.txt")
    fake_valid = read_list(list_dir / "FR_valid_list.txt")
    real_train = read_list(list_dir / "FR_auth_train_list.txt")
    real_valid = read_list(list_dir / "FR_auth_valid_list.txt")

    if args.limit > 0:
        fake_train = fake_train[: args.limit]
        fake_valid = fake_valid[: args.limit]
        real_train = real_train[: args.limit]
        real_valid = real_valid[: args.limit]

    validate_paths(root, fake_train + fake_valid, real_train + real_valid)
    write_doc_lite_list(out_dir / args.train_out, fake_train, real_train)
    write_doc_lite_list(out_dir / args.valid_out, fake_valid, real_valid)

    converted = 0
    skipped = 0
    if not args.no_convert:
        if args.convert_all:
            fake_names = sorted(path.name for path in (root / "SegmentationFake").glob(f"*{NPZ_EXT}"))
            fake_names = [Path(name).with_suffix(IMAGE_EXT).name for name in fake_names]
        else:
            fake_names = sorted(set(fake_train + fake_valid))
        converted, skipped = convert_masks(root, fake_names, skip_existing=not args.overwrite)

    print(f"root: {root}")
    print(f"train list: {out_dir / args.train_out} fake={len(fake_train)} real={len(real_train)}")
    print(f"valid list: {out_dir / args.valid_out} fake={len(fake_valid)} real={len(real_valid)}")
    print(f"mask pngs: converted={converted} skipped={skipped}")


if __name__ == "__main__":
    main()
