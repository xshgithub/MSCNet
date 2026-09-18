import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from PIL import JpegImagePlugin, PdfImagePlugin  # noqa: F401 - register PDF/JPEG savers


MODALITIES = ("SLA", "SSH", "SST", "CHL")
CANVAS = (2200, 980)
PANEL = 170
BG = (247, 249, 250)
INK = (28, 38, 48)
MUTED = (103, 116, 128)
LINE = (199, 208, 216)
BLUE = (32, 108, 181)
TEAL = (26, 153, 146)
ORANGE = (228, 123, 54)
RED = (208, 70, 63)
GREEN = (65, 155, 92)


def font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def load_feature(path, size=PANEL):
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


def rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def paste_panel(canvas, image, xy, label, border=LINE):
    x, y = xy
    shadow = Image.new("RGBA", (PANEL + 16, PANEL + 16), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    rounded_rect(sd, (8, 8, PANEL + 8, PANEL + 8), 12, (0, 0, 0, 38))
    shadow = shadow.filter(ImageFilter.GaussianBlur(7))
    canvas.alpha_composite(shadow, (x - 8, y - 8))

    mask = Image.new("L", (PANEL, PANEL), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, PANEL, PANEL), radius=12, fill=255)
    canvas.paste(image, (x, y), mask)

    draw = ImageDraw.Draw(canvas)
    rounded_rect(draw, (x, y, x + PANEL, y + PANEL), 12, None, border, 2)
    draw.text((x + 10, y + 9), label, fill=(255, 255, 255), font=font(20, True))


def draw_arrow(draw, start, end, color=INK, width=4):
    draw.line((start, end), fill=color, width=width)
    sx, sy = start
    ex, ey = end
    angle = np.arctan2(ey - sy, ex - sx)
    size = 14
    points = [
        (ex, ey),
        (ex - size * np.cos(angle - 0.45), ey - size * np.sin(angle - 0.45)),
        (ex - size * np.cos(angle + 0.45), ey - size * np.sin(angle + 0.45)),
    ]
    draw.polygon(points, fill=color)


def draw_dashed_ellipse(draw, box, color, width=2, dash=9):
    x0, y0, x1, y1 = box
    for angle in range(0, 360, dash * 2):
        draw.arc((x0, y0, x1, y1), angle, angle + dash, fill=color, width=width)


def draw_misalignment_overlay(draw, x, y, modality_idx):
    centers = [
        (x + 70, y + 76, BLUE),
        (x + 98, y + 66, TEAL),
        (x + 116, y + 104, ORANGE),
        (x + 82, y + 119, RED),
    ]
    ref_cx, ref_cy = x + 88, y + 92
    draw_dashed_ellipse(draw, (ref_cx - 43, ref_cy - 31, ref_cx + 43, ref_cy + 31),
                        (238, 242, 245), width=3)
    cx, cy, color = centers[modality_idx]
    draw.ellipse((cx - 44, cy - 30, cx + 44, cy + 30), outline=color, width=5)
    draw.line((ref_cx, ref_cy, cx, cy), fill=(245, 246, 248), width=3)
    draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=color)
    draw.ellipse((ref_cx - 4, ref_cy - 4, ref_cx + 4, ref_cy + 4), outline=(245, 246, 248), width=2)


def draw_alignment_overlay(draw, x, y):
    cx, cy = x + PANEL // 2, y + PANEL // 2
    colors = (BLUE, TEAL, ORANGE, RED)
    offsets = ((-34, -24), (32, -22), (35, 30), (-31, 27))
    for dx, dy in offsets:
        draw_arrow(draw, (cx + dx, cy + dy), (cx, cy), color=(78, 91, 105), width=3)
    for idx, color in enumerate(colors):
        pad = 34 + idx * 5
        draw.ellipse((x + pad, y + pad, x + PANEL - pad, y + PANEL - pad), outline=color, width=3)
    draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=GREEN)


def write_centered(draw, text, center_x, y, fnt, fill=INK):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((center_x - (bbox[2] - bbox[0]) / 2, y), text, fill=fill, font=fnt)


def draw_motivation(feature_root, out_path):
    feature_root = Path(feature_root)
    out_path = Path(out_path)
    canvas = Image.new("RGBA", CANVAS, BG + (255,))
    draw = ImageDraw.Draw(canvas)

    title_font = font(38, True)
    head_font = font(27, True)
    label_font = font(20)
    small_font = font(17)

    draw.text((72, 48), "Motivation: Cross-modal spatial mismatch in mesoscale eddy detection",
              fill=INK, font=title_font)
    draw.text((74, 96),
              "Different ocean dynamics shift vortex boundaries and centers across modalities; implicit shallow fusion weakens spatial correspondence.",
              fill=MUTED, font=label_font)

    col1_x, col2_x, col3_x = 78, 810, 1540
    top_y = 205

    write_centered(draw, "(a) Multi-modal observations", col1_x + 280, 160, head_font)
    write_centered(draw, "(b) Explicit structure alignment", col2_x + 280, 160, head_font)
    write_centered(draw, "(c) Spatially coherent fusion", col3_x + 280, 160, head_font)

    before_positions = [
        (col1_x, top_y),
        (col1_x + PANEL + 36, top_y),
        (col1_x, top_y + PANEL + 54),
        (col1_x + PANEL + 36, top_y + PANEL + 54),
    ]
    for idx, (modality, pos) in enumerate(zip(MODALITIES, before_positions)):
        image = load_feature(feature_root / f"before_{modality.lower()}.png")
        paste_panel(canvas, image, pos, modality)
        draw_misalignment_overlay(draw, *pos, idx)

    draw.text((col1_x + 15, 612), "same eddy, shifted geometry", fill=RED, font=label_font)
    draw.text((col1_x + 15, 644), "boundary / center mismatch", fill=MUTED, font=small_font)

    after_positions = [
        (col2_x, top_y),
        (col2_x + PANEL + 36, top_y),
        (col2_x, top_y + PANEL + 54),
        (col2_x + PANEL + 36, top_y + PANEL + 54),
    ]
    for modality, pos in zip(MODALITIES, after_positions):
        image = load_feature(feature_root / f"after_{modality.lower()}.png")
        paste_panel(canvas, image, pos, modality)
        draw_alignment_overlay(draw, *pos)

    draw.text((col2_x + 15, 612), "DWT-enhanced spatial cues", fill=TEAL, font=label_font)
    draw.text((col2_x + 15, 644), "features encouraged toward common structure", fill=MUTED, font=small_font)

    fused = load_feature(feature_root / "fused.png", size=300)
    fused_x, fused_y = col3_x + 130, top_y + 55
    shadow = Image.new("RGBA", (332, 332), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    rounded_rect(sd, (16, 16, 316, 316), 18, (0, 0, 0, 42))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    canvas.alpha_composite(shadow, (fused_x - 16, fused_y - 16))
    mask = Image.new("L", (300, 300), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, 300, 300), radius=18, fill=255)
    canvas.paste(fused, (fused_x, fused_y), mask)
    rounded_rect(draw, (fused_x, fused_y, fused_x + 300, fused_y + 300), 18, None, GREEN, 3)
    draw.ellipse((fused_x + 86, fused_y + 78, fused_x + 214, fused_y + 206), outline=GREEN, width=5)
    draw.ellipse((fused_x + 144, fused_y + 136, fused_x + 156, fused_y + 148), fill=GREEN)
    draw.text((fused_x + 18, fused_y + 16), "FUSED", fill=(255, 255, 255), font=font(24, True))

    draw.text((col3_x + 78, 612), "aligned correspondence", fill=GREEN, font=label_font)
    draw.text((col3_x + 78, 644), "better extraction of complementary evidence", fill=MUTED, font=small_font)

    draw_arrow(draw, (col1_x + 464, 392), (col2_x - 54, 392), color=(64, 77, 91), width=4)
    draw_arrow(draw, (col2_x + 464, 392), (col3_x - 54, 392), color=(64, 77, 91), width=4)

    problem_box = (74, 748, 2128, 888)
    rounded_rect(draw, problem_box, 18, (255, 255, 255), (218, 225, 231), 2)
    draw.text((110, 779), "Problem", fill=RED, font=font(24, True))
    draw.text((242, 782),
              "Implicit shallow fusion lacks explicit constraints on vortex spatial consistency, so cross-modal centers and boundaries are hard to match.",
              fill=INK, font=label_font)
    draw.text((110, 830), "Motivation", fill=TEAL, font=font(24, True))
    draw.text((242, 833),
              "Use structure-aware alignment to build reliable spatial correspondence before extracting complementary multimodal information.",
              fill=INK, font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = canvas.convert("RGB")
    rgb.save(out_path, quality=96)
    rgb.save(out_path.with_suffix(".pdf"))
    return out_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", default="output/feature_vis/stage1")
    parser.add_argument("--out", default="figure/motivation_multimodal_alignment.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = draw_motivation(args.feature_root, args.out)
    print(path.resolve())
