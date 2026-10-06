from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class TransformConfig:
    image_size: int = 512
    train: bool = True
    heavy_degrade: bool = False
    hard_negative: bool = False
    hard_negative_only_real: bool = True
    text_edge_hard_negative_prob: float = 0.35
    security_pattern_hard_negative_prob: float = 0.25
    keep_aspect_ratio: bool = True
    pad_value: int = 255


def _resize_image(image: Image.Image, size: int, is_mask: bool = False) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BICUBIC
    return image.resize((size, size), resample=resample)


def get_letterbox_params(original_size: tuple[int, int], size: int) -> dict[str, int | float]:
    width, height = original_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {original_size}")
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2
    return {
        "scale": scale,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": size - resized_width - pad_left,
        "pad_bottom": size - resized_height - pad_top,
    }


def _letterbox_image(
    image: Image.Image,
    size: int,
    is_mask: bool = False,
    pad_value: int = 255,
) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BICUBIC
    params = get_letterbox_params(image.size, size)
    resized_size = (int(params["resized_width"]), int(params["resized_height"]))
    resized = image.resize(resized_size, resample=resample)
    fill = 0 if is_mask else pad_value
    if image.mode == "RGB":
        fill = (int(fill), int(fill), int(fill))
    canvas = Image.new(image.mode, (size, size), color=fill)
    canvas.paste(resized, (int(params["pad_left"]), int(params["pad_top"])))
    return canvas


def _jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _rotate_pair(
    image: Image.Image,
    mask: Image.Image | None,
    max_angle: float = 4.0,
) -> tuple[Image.Image, Image.Image | None]:
    angle = random.uniform(-max_angle, max_angle)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    if mask is not None:
        mask = mask.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=0)
    return image, mask


def _perspective_coeffs(dst: list[tuple[float, float]], src: list[tuple[float, float]]) -> list[float]:
    import numpy as np

    matrix = []
    target = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        target.extend([u, v])
    return np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(target, dtype=np.float64)).tolist()


def _perspective_pair(
    image: Image.Image,
    mask: Image.Image | None,
    max_warp: float = 0.045,
) -> tuple[Image.Image, Image.Image | None]:
    width, height = image.size
    if width < 4 or height < 4:
        return image, mask

    dx = width * max_warp
    dy = height * max_warp
    src = [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))]
    dst = [
        (random.uniform(0.0, dx), random.uniform(0.0, dy)),
        (width - random.uniform(0.0, dx), random.uniform(0.0, dy)),
        (width - random.uniform(0.0, dx), height - random.uniform(0.0, dy)),
        (random.uniform(0.0, dx), height - random.uniform(0.0, dy)),
    ]
    try:
        coeffs = _perspective_coeffs(dst, src)
    except Exception:
        return image, mask
    image = image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(255, 255, 255))
    if mask is not None:
        mask = mask.transform(mask.size, Image.PERSPECTIVE, coeffs, Image.NEAREST, fillcolor=0)
    return image, mask


def _color_jitter(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.30))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.80, 1.20))
    if random.random() < 0.4:
        image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.60, 1.80))
    return image


def _low_res_roundtrip(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = random.uniform(0.35, 0.75)
    small_size = (max(8, int(width * scale)), max(8, int(height * scale)))
    return image.resize(small_size, resample=Image.BICUBIC).resize((width, height), resample=Image.BICUBIC)


def _local_shadow(image: Image.Image, strength: float = 0.35) -> Image.Image:
    import numpy as np

    arr = np.asarray(image).astype(np.float32)
    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    angle = random.uniform(0.0, np.pi)
    gradient = xx * np.cos(angle) + yy * np.sin(angle)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    shadow = np.clip((gradient - random.uniform(0.25, 0.65)) * random.uniform(2.0, 4.0), 0.0, 1.0)
    shadow = shadow[..., None] * random.uniform(0.15, strength)
    return Image.fromarray(np.clip(arr * (1.0 - shadow), 0, 255).astype(np.uint8))


def _local_highlight(image: Image.Image, strength: float = 0.60) -> Image.Image:
    import numpy as np

    arr = np.asarray(image).astype(np.float32)
    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cx = random.uniform(0.1 * width, 0.9 * width)
    cy = random.uniform(0.1 * height, 0.9 * height)
    rx = random.uniform(0.12 * width, 0.35 * width)
    ry = random.uniform(0.08 * height, 0.28 * height)
    blob = np.exp(-(((xx - cx) / (rx + 1e-6)) ** 2 + ((yy - cy) / (ry + 1e-6)) ** 2))
    alpha = blob[..., None] * random.uniform(0.20, strength)
    return Image.fromarray(np.clip(arr * (1.0 - alpha) + 255.0 * alpha, 0, 255).astype(np.uint8))


def _moire_pattern(image: Image.Image, strength: float = 0.06) -> Image.Image:
    import numpy as np

    arr = np.asarray(image).astype(np.float32) / 255.0
    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    angle = random.uniform(0.0, np.pi)
    freq = random.uniform(0.08, 0.22)
    phase = random.uniform(0.0, 2.0 * np.pi)
    pattern = np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * freq + phase)
    arr = np.clip(arr + pattern[..., None] * random.uniform(0.015, strength), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _print_scan_texture(image: Image.Image, strength: float = 0.04) -> Image.Image:
    import numpy as np

    arr = np.asarray(image).astype(np.float32) / 255.0
    height = arr.shape[0]
    line = np.random.normal(0.0, random.uniform(0.005, strength), (height, 1, 1))
    grain = np.random.normal(0.0, random.uniform(0.004, strength * 0.75), arr.shape)
    arr = np.clip(arr + line + grain, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _scanline_noise(image: Image.Image, strength: float = 0.05) -> Image.Image:
    import numpy as np

    arr = np.asarray(image).astype(np.float32) / 255.0
    h = arr.shape[0]
    line = np.sin(np.linspace(0, np.pi * random.uniform(12, 36), h))
    line = line[:, None, None] * strength
    arr = np.clip(arr + line, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _gaussian_noise(image: Image.Image, sigma: float = 0.025) -> Image.Image:
    import numpy as np

    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.clip(arr + np.random.normal(0.0, sigma, arr.shape), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _text_edge_hard_negative(image: Image.Image, strength: float = 0.18) -> Image.Image:
    import numpy as np

    rgb = image.convert("RGB")
    edge = rgb.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=0.35))
    edge_arr = np.asarray(edge).astype(np.float32) / 255.0
    edge_arr = np.clip((edge_arr - 0.10) / 0.45, 0.0, 1.0)
    arr = np.asarray(rgb).astype(np.float32)
    sign = -1.0 if random.random() < 0.65 else 1.0
    arr = arr + sign * edge_arr[..., None] * random.uniform(20.0, 70.0) * strength
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    if random.random() < 0.7:
        out = out.filter(
            ImageFilter.UnsharpMask(
                radius=random.uniform(0.6, 1.2),
                percent=random.randint(90, 180),
                threshold=random.randint(2, 8),
            )
        )
    if random.random() < 0.5:
        out = ImageEnhance.Contrast(out).enhance(random.uniform(1.03, 1.16))
    return out


def _security_pattern_hard_negative(image: Image.Image, strength: float = 0.055) -> Image.Image:
    import numpy as np

    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = np.zeros((height, width), dtype=np.float32)
    for _ in range(random.randint(2, 4)):
        angle = random.uniform(0.0, np.pi)
        freq = random.uniform(0.035, 0.095)
        phase = random.uniform(0.0, 2.0 * np.pi)
        wave = np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * freq + phase)
        pattern += wave
    pattern = pattern / max(1.0, float(np.max(np.abs(pattern))))
    tint = np.asarray(
        random.choice(
            [
                (0.10, 0.18, 0.35),
                (0.10, 0.32, 0.25),
                (0.35, 0.20, 0.08),
                (0.30, 0.12, 0.30),
            ]
        ),
        dtype=np.float32,
    )
    alpha = np.clip((pattern + 1.0) * 0.5, 0.0, 1.0)[..., None] * random.uniform(0.35, 1.0) * strength
    arr = arr * (1.0 - alpha) + tint.reshape(1, 1, 3) * alpha
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")


class DocLiteTransform:
    def __init__(self, cfg: TransformConfig):
        self.cfg = cfg

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
        label: float | int | None = None,
    ) -> tuple[Image.Image, Image.Image | None]:
        image = image.convert("RGB")
        if mask is not None:
            mask = mask.convert("L")

        if self.cfg.train:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if mask is not None:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.2:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                if mask is not None:
                    mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.5:
                k = random.randint(0, 3)
                if k:
                    image = image.rotate(90 * k, expand=True)
                    if mask is not None:
                        mask = mask.rotate(90 * k, expand=True)

            if self.cfg.heavy_degrade:
                # Geometry changes must be applied to image and mask together.
                if random.random() < 0.35:
                    image, mask = _rotate_pair(image, mask)
                if random.random() < 0.35:
                    image, mask = _perspective_pair(image, mask)

                # These simulate normal capture degradation; the mask is unchanged.
                if random.random() < 0.50:
                    image = _color_jitter(image)
                if random.random() < 0.40:
                    image = _low_res_roundtrip(image)
                if random.random() < 0.35:
                    image = _local_shadow(image)
                if random.random() < 0.30:
                    image = _local_highlight(image)
                if random.random() < 0.25:
                    image = _moire_pattern(image)
                if random.random() < 0.25:
                    image = _print_scan_texture(image)
                if random.random() < 0.7:
                    image = _jpeg_recompress(image, quality=random.randint(20, 60))
                if random.random() < 0.35:
                    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.6)))
                if random.random() < 0.35:
                    image = _gaussian_noise(image, sigma=random.uniform(0.01, 0.04))
                if random.random() < 0.25:
                    image = _scanline_noise(image, strength=random.uniform(0.02, 0.07))
            else:
                if random.random() < 0.25:
                    image = _jpeg_recompress(image, quality=random.randint(60, 95))

            is_real = label is not None and float(label) < 0.5
            use_hard_negative = self.cfg.hard_negative and (is_real or not self.cfg.hard_negative_only_real)
            if use_hard_negative:
                if random.random() < self.cfg.text_edge_hard_negative_prob:
                    image = _text_edge_hard_negative(image)
                if random.random() < self.cfg.security_pattern_hard_negative_prob:
                    image = _security_pattern_hard_negative(image)

        if self.cfg.keep_aspect_ratio:
            image = _letterbox_image(
                image,
                self.cfg.image_size,
                is_mask=False,
                pad_value=self.cfg.pad_value,
            )
            if mask is not None:
                mask = _letterbox_image(mask, self.cfg.image_size, is_mask=True)
        else:
            image = _resize_image(image, self.cfg.image_size, is_mask=False)
            if mask is not None:
                mask = _resize_image(mask, self.cfg.image_size, is_mask=True)
        return image, mask


def build_transform(cfg: dict[str, Any], train: bool) -> DocLiteTransform:
    return DocLiteTransform(
        TransformConfig(
            image_size=int(cfg.get("image_size", 512)),
            train=train,
            heavy_degrade=bool(cfg.get("heavy_degrade", False)),
            hard_negative=bool(cfg.get("hard_negative", False)),
            hard_negative_only_real=bool(cfg.get("hard_negative_only_real", True)),
            text_edge_hard_negative_prob=float(cfg.get("text_edge_hard_negative_prob", 0.35)),
            security_pattern_hard_negative_prob=float(cfg.get("security_pattern_hard_negative_prob", 0.25)),
            keep_aspect_ratio=bool(cfg.get("keep_aspect_ratio", True)),
            pad_value=int(cfg.get("pad_value", 255)),
        )
    )
