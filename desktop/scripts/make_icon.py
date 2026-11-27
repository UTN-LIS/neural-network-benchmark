"""Generate the desktop icon from code so the build is reproducible.

Run from the repository root:

    python desktop/scripts/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

CANVAS = 1024
BACKGROUND = (11, 18, 32, 255)
BORDER = (30, 58, 95, 255)
CYAN = (56, 189, 248, 255)
AMBER = (245, 158, 11, 255)
LINE = (56, 189, 248, 110)


def main() -> int:
    image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (32, 32, CANVAS - 32, CANVAS - 32),
        radius=220,
        fill=BACKGROUND,
        outline=BORDER,
        width=16,
    )

    left = [(232, 330), (232, 512), (232, 694)]
    middle = [(512, 280), (512, 512), (512, 744)]
    right = [(792, 350), (792, 512), (792, 674)]

    for source in left + middle:
        targets = right if source in middle else middle
        for target in targets:
            draw.line((source, target), fill=LINE, width=14)

    for x, y in left:
        draw.ellipse((x - 46, y - 46, x + 46, y + 46), fill=CYAN)
    for x, y in middle:
        draw.ellipse((x - 40, y - 40, x + 40, y + 40), fill=CYAN)
    for x, y in right:
        draw.ellipse((x - 46, y - 46, x + 46, y + 46), fill=AMBER)

    out_dir = Path(__file__).resolve().parents[1] / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "icon.png"
    ico_path = out_dir / "icon.ico"

    image.save(png_path)
    image.save(
        ico_path,
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"wrote {png_path}")
    print(f"wrote {ico_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
