from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path


WEIGHTS = {
    "mit_b2": {
        "url": "https://raw.githubusercontent.com/grip-unina/TruFor/main/TruFor_train_test/pretrained_models/segformers/mit_b2.pth",
        "path": Path("weights/segformers/mit_b2.pth"),
        "sha256": "ced22617efb7bae3c34ad0a80f20a9b8afb4d27368cb0835a23456baa9d0e092",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, force: bool = False) -> Path:
    item = WEIGHTS[name]
    destination = Path(item["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        if sha256(destination) == item["sha256"]:
            print(f"verified: {destination}")
            return destination
        raise RuntimeError(f"Checksum mismatch for existing file: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(
        item["url"], headers={"User-Agent": "ForenID-Net weight downloader"}
    )
    print(f"downloading {name} -> {destination}")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual = sha256(temporary)
    if actual != item["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {name}: expected {item['sha256']}, got {actual}"
        )
    temporary.replace(destination)
    print(f"verified: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download verified pretrained weights.")
    parser.add_argument("names", nargs="*", default=list(WEIGHTS), choices=WEIGHTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        download(name, force=args.force)


if __name__ == "__main__":
    main()
