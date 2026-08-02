#!/usr/bin/env python3
"""Generate brand and marketing assets.

Everything the site ships is produced here rather than pulled from a stock
library, for two reasons: a real patient radiograph must never appear on a
public page, and an image whose licence we cannot verify must never ship at
all. Re-run with `make assets` after changing the brand colour.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "dentist_ai" / "static"
IMAGES = STATIC / "images"
ICONS = STATIC / "icons"

BRAND = "#3b7fd4"


def _tooth(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    width: float,
    height: float,
    brightness: int,
    *,
    roots_down: bool,
) -> None:
    """One crown-plus-roots silhouette.

    ``roots_down`` flips the whole shape for the lower arch, where roots point
    away from the occlusal plane in the opposite direction.
    """
    direction = 1.0 if roots_down else -1.0

    crown_near = cy - direction * height * 0.5
    crown_far = cy + direction * height * 0.05
    draw.rounded_rectangle(
        (cx - width / 2, min(crown_near, crown_far), cx + width / 2, max(crown_near, crown_far)),
        radius=width * 0.28,
        fill=brightness,
    )

    root_width = width * 0.3
    for offset in (-width * 0.22, width * 0.22):
        draw.polygon(
            [
                (cx + offset - root_width / 2, cy),
                (cx + offset + root_width / 2, cy),
                (cx + offset, cy + direction * height * 0.55),
            ],
            fill=int(brightness * 0.86),
        )


def render_radiograph(*, size: tuple[int, int] = (1200, 900), seed: int = 20260728) -> Image.Image:
    """Render a synthetic panoramic radiograph.

    Shared with ``scripts/seed.py`` so demo data and the marketing hero come
    from one generator — varying only the seed — rather than two that drift.
    """
    rng = random.Random(seed)
    width, height = size
    image = Image.new("L", size, color=14)
    draw = ImageDraw.Draw(image)

    # Soft-tissue haze. Broad and low-contrast: in a real panoramic the
    # background is a gradient, not a hard silhouette.
    draw.ellipse((-width * 0.25, -height * 0.35, width * 1.25, height * 1.15), fill=30)
    draw.ellipse((width * 0.05, height * 0.1, width * 0.95, height * 1.3), fill=24)

    # Ramus of the mandible on each side.
    for side in (0.13, 0.87):
        draw.ellipse(
            (
                width * side - width * 0.055,
                height * 0.42,
                width * side + width * 0.055,
                height * 1.05,
            ),
            fill=62,
        )

    # Inferior border of the mandible, sweeping below the lower arch.
    draw.arc(
        (width * 0.09, height * 0.42, width * 0.91, height * 1.28),
        start=5,
        end=175,
        fill=104,
        width=int(height * 0.035),
    )

    # Both arches follow the same shallow "smile": molars sit higher than the
    # incisors, which is what a panoramic projection produces.
    for arch_index, (base_y, curvature, scale) in enumerate(
        ((height * 0.44, height * 0.09, 1.0), (height * 0.63, height * 0.10, 0.94))
    ):
        count = 16
        for index in range(count):
            t = index / (count - 1)
            cx = width * 0.16 + t * width * 0.68
            # Molars sit lower/higher at the ends of the arch than incisors.
            cy = base_y + curvature * math.sin(math.pi * t)
            is_molar = t < 0.22 or t > 0.78
            tooth_width = width * (0.045 if is_molar else 0.034) * scale
            tooth_height = height * (0.1 if is_molar else 0.125) * scale
            brightness = rng.randint(178, 232)
            # Upper arch: roots point up. Lower arch: roots point down.
            roots_down = arch_index == 1
            _tooth(draw, cx, cy, tooth_width, tooth_height, brightness, roots_down=roots_down)

            # Occasional bright restoration, so the image reads as clinical.
            if rng.random() < 0.22:
                inset = tooth_width * 0.22
                direction = 1.0 if roots_down else -1.0
                near = cy - direction * tooth_height * 0.3
                far = cy - direction * tooth_height * 0.08
                draw.ellipse(
                    (cx - inset, min(near, far), cx + inset, max(near, far)),
                    fill=252,
                )

    image = image.filter(ImageFilter.GaussianBlur(radius=1.6))

    # Film grain: a perfectly smooth render looks synthetic.
    noise = Image.effect_noise(size, 12).point(lambda value: int(value * 0.35))
    image = Image.blend(image, Image.composite(noise, image, noise), 0.12)

    # Vignette.
    vignette = Image.new("L", size, 0)
    ImageDraw.Draw(vignette).ellipse(
        (-width * 0.15, -height * 0.15, width * 1.15, height * 1.15), fill=255
    )
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=width * 0.08))
    image = Image.composite(image, Image.new("L", size, 6), vignette)

    return image.convert("RGB")


def generate_radiograph(path: Path, size: tuple[int, int] = (1200, 900)) -> None:
    image = render_radiograph(size=size)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=84, optimize=True, progressive=True)
    print(f"wrote {path.relative_to(ROOT)}")


FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="7" fill="{brand}"/>
  <path d="M11 7c-2.8 0-4.6 2.1-4.6 5.2 0 2.4.7 4.4 1.6 7 .8 2.3 1.3 5.8 3.1 5.8 1.5 0 1.6-2.6 2.3-4.6.4-1.1.9-1.7 2.6-1.7s2.2.6 2.6 1.7c.7 2 .8 4.6 2.3 4.6 1.8 0 2.3-3.5 3.1-5.8.9-2.6 1.6-4.6 1.6-7C25.6 9.1 23.8 7 21 7c-2.1 0-3.4 1.1-5 1.1S13.1 7 11 7Z"
        fill="#fff"/>
</svg>
"""


def generate_icons() -> None:
    ICONS.mkdir(parents=True, exist_ok=True)
    (ICONS / "favicon.svg").write_text(FAVICON_SVG.format(brand=BRAND), encoding="utf-8")
    print("wrote src/dentist_ai/static/icons/favicon.svg")

    # Raster fallbacks for platforms that ignore SVG favicons.
    for name, size in (("apple-touch-icon.png", 180), ("icon-512.png", 512)):
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        radius = int(size * 0.22)
        draw.rounded_rectangle((0, 0, size, size), radius=radius, fill=BRAND)

        # A simple, legible glyph — fine detail disappears at 16 px anyway.
        unit = size / 32
        draw.rounded_rectangle(
            (10 * unit, 8 * unit, 15 * unit, 24 * unit), radius=2.4 * unit, fill="white"
        )
        draw.rounded_rectangle(
            (17 * unit, 8 * unit, 22 * unit, 24 * unit), radius=2.4 * unit, fill="white"
        )
        canvas.save(ICONS / name, format="PNG", optimize=True)
        print(f"wrote src/dentist_ai/static/icons/{name}")


def generate_og_image(path: Path) -> None:
    """Open Graph card, 1200x630."""
    size = (1200, 630)
    image = Image.new("RGB", size, (13, 17, 23))
    draw = ImageDraw.Draw(image)

    glow = Image.new("RGB", size, (13, 17, 23))
    ImageDraw.Draw(glow).ellipse((250, -260, 1150, 500), fill=(28, 56, 104))
    image = Image.blend(image, glow.filter(ImageFilter.GaussianBlur(140)), 0.85)

    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((80, 72, 152, 144), radius=18, fill=BRAND)
    draw.text((180, 96), "Dentist-AI", fill="white")
    draw.text(
        (84, 300),
        "ИИ-анализ дентальных снимков\n31 класс находок с оценкой уверенности",
        fill=(200, 210, 225),
        spacing=14,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=88, optimize=True)
    print(f"wrote {path.relative_to(ROOT)}")


WEBMANIFEST = """{
  "name": "Dentist-AI",
  "short_name": "Dentist-AI",
  "description": "ИИ-анализ дентальных рентгеновских снимков",
  "start_url": "/app",
  "scope": "/",
  "display": "standalone",
  "background_color": "#0d1117",
  "theme_color": "#0d1117",
  "lang": "ru",
  "icons": [
    { "src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png",
      "purpose": "any maskable" },
    { "src": "/static/icons/favicon.svg", "sizes": "any", "type": "image/svg+xml" }
  ]
}
"""


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    generate_radiograph(IMAGES / "radiograph-sample.jpg")
    generate_og_image(IMAGES / "og-card.jpg")
    generate_icons()
    (STATIC / "site.webmanifest").write_text(WEBMANIFEST, encoding="utf-8")
    print("wrote src/dentist_ai/static/site.webmanifest")


if __name__ == "__main__":
    main()
