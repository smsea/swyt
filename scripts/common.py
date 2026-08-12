# -*- coding: utf-8 -*-
"""
다미앤마케팅 전자책 광고 소재 — 공용 렌더링 모듈

한글 텍스트는 전부 여기서 PIL로 렌더링해 굽는다.
ffmpeg drawtext 는 쓰지 않는다 (폰트·인코딩 사고 원천 차단).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

# ── 브랜드 ────────────────────────────────────────────────────────────────
GOLD = (255, 209, 102)      # #FFD166 — 강조어 "구조" 전용
WHITE = (255, 255, 255)
BRAND = "다미앤마케팅"
HIGHLIGHT = "구조"          # 이 어절만 골드. 다른 단어에 쓰지 말 것.

# ── 규격 ──────────────────────────────────────────────────────────────────
# safe: (상단, 하단) 비율. 9:16 은 상 15% / 하 21%, 1:1 은 상하 7%.
# 4:5 는 지시문에 명시가 없어 두 값 사이인 8% 로 둔다 (RATIOS 에서 조정 가능).
RATIOS = {
    "1x1":  dict(size=(1080, 1080), safe=(0.07, 0.07)),
    "4x5":  dict(size=(1080, 1350), safe=(0.08, 0.08)),
    "9x16": dict(size=(1080, 1920), safe=(0.15, 0.21)),
}

SIDE_MARGIN = 0.085         # 좌우 여백 각 8.5%
JPEG_QUALITY = 92

FONT_PATH = SCRIPTS / "Pretendard-Black.otf"

# 헤드라인 기준 크기 (캔버스 너비 대비). 여백을 넘으면 자동 축소한다.
HEADLINE_SCALE = 0.082
LINE_SPACING = 1.24
BRAND_SCALE = 0.030
BRAND_GAP = 0.55            # 헤드라인 대비 브랜드 위 간격 (헤드라인 글자 크기 배수)

FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", shutil.which("ffprobe") or "ffprobe")


# ── 카피 (확정 3비트 — 변경 금지) ─────────────────────────────────────────
BEATS = [
    dict(id="b1", role="문제 제기", lines=["매일 글 쓰는데", "예약은 왜 안 늘까"]),
    dict(id="b2", role="인식 전환", lines=["글쓰기가 아니라", "구조의 문제였습니다"]),
    dict(id="b3", role="해결 제시", lines=["검색부터 예약까지", "구조로 완성합니다"]),
]


# ── 이미지 로딩 ───────────────────────────────────────────────────────────
def load_image(path) -> Image.Image:
    """폰 촬영 JPG 는 반드시 exif_transpose 를 먼저 적용한다. 안 하면 90도 눕는다."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# ── 스마트 크롭 ───────────────────────────────────────────────────────────
def _energy_map(img: Image.Image, cols: int = 64, rows: int = 64) -> np.ndarray:
    """엣지 + 채도 기반 관심도 맵. 책 표지처럼 대비가 큰 영역이 높게 잡힌다."""
    small = img.convert("RGB").resize((cols, rows), Image.BILINEAR)
    gray = small.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = np.asarray(gray, dtype=np.float32)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    sat = hsv[:, :, 1]
    e = edges / (edges.max() or 1.0) + 0.35 * (sat / 255.0)
    return e


def smart_crop_box(
    img: Image.Image,
    target_w: int,
    target_h: int,
    anchor: tuple[float, float] | None = None,
    text_zone: float = 0.34,
) -> tuple[int, int, int, int]:
    """
    책이 잘리지 않도록 크롭 기준점을 잡는다. 단순 중앙 크롭이 아니다.

    anchor 가 주어지면 (원본 기준 0~1 좌표) 그 지점을 중심으로 삼는다.
    없으면 관심도 맵에서, 텍스트가 올라갈 하단 영역을 제외한 상단부에
    피사체가 오도록 오프셋을 고른다.
    """
    W, H = img.size
    ar_t = target_w / target_h
    ar_s = W / H

    if ar_s > ar_t:                      # 원본이 더 넓다 → 가로를 자른다
        cw, ch = int(round(H * ar_t)), H
        span, axis = W - cw, "x"
    else:                                # 원본이 더 높다 → 세로를 자른다
        cw, ch = W, int(round(W / ar_t))
        span, axis = H - ch, "y"

    if span <= 0:
        return (0, 0, cw, ch)

    if anchor is not None:
        fx, fy = anchor
        off = int(round(fx * W - cw / 2)) if axis == "x" else int(round(fy * H - ch / 2))
    else:
        # 피사체(책)의 '분포 전체'를 담도록 잡는다. 가장 진한 한 점에 맞추면
        # 표지 일부만 남고 잘린다 — 무게 중심이 아니라 분포 구간의 중앙을 쓴다.
        e = _energy_map(img)
        prof = e.sum(axis=0) if axis == "x" else e.sum(axis=1)
        total = float(prof.sum())
        if total <= 0:
            off = span // 2
        else:
            cum = np.cumsum(prof) / total
            lo = int(np.searchsorted(cum, 0.05))
            hi = int(np.searchsorted(cum, 0.95))
            n = len(prof)
            mid = ((lo + hi) / 2.0) / max(n - 1, 1)          # 0~1
            extent = (W, cw) if axis == "x" else (H, ch)
            off = int(round(mid * extent[0] - extent[1] / 2))
            if axis == "y":
                # 하단은 텍스트 자리 → 창을 조금 내려 피사체를 위쪽에 앉힌다
                off += int(round(ch * text_zone * 0.18))

    off = max(0, min(span, off))
    return (off, 0, off + cw, ch) if axis == "x" else (0, off, cw, off + ch)


def crop_to(img: Image.Image, w: int, h: int, anchor=None) -> Image.Image:
    box = smart_crop_box(img, w, h, anchor)
    return img.crop(box).resize((w, h), Image.LANCZOS)


# ── 텍스트 렌더링 ─────────────────────────────────────────────────────────
def _font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), px)


def _measure(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return int(draw.textlength(text, font=font))


def fit_font(draw, lines: list[str], max_w: int, start_px: int) -> ImageFont.FreeTypeFont:
    """긴 줄이 여백을 넘으면 들어올 때까지 폰트를 자동 축소한다."""
    px = start_px
    while px > 12:
        f = _font(px)
        if all(_measure(draw, ln, f) <= max_w for ln in lines):
            return f
        px -= 2
    return _font(12)


def _draw_highlighted(draw, xy, text: str, font, base=WHITE, key=HIGHLIGHT, key_color=GOLD):
    """본문은 흰색, 강조어(구조)만 골드로 이어 그린다."""
    x, y = xy
    rest = text
    while rest:
        i = rest.find(key)
        if i < 0:
            draw.text((x, y), rest, font=font, fill=base)
            return
        if i:
            head = rest[:i]
            draw.text((x, y), head, font=font, fill=base)
            x += _measure(draw, head, font)
        draw.text((x, y), key, font=font, fill=key_color)
        x += _measure(draw, key, font)
        rest = rest[i + len(key):]


def bottom_scrim(size: tuple[int, int], top_frac: float = 0.46, strength: int = 205) -> Image.Image:
    """
    가독성 확보용 하단 그라디언트 스크림. 사진 상단은 건드리지 않는다.
    top_frac 위쪽은 완전 투명.
    """
    w, h = size
    ramp = np.zeros(h, dtype=np.float32)
    start = int(h * top_frac)
    if start < h:
        t = np.linspace(0.0, 1.0, h - start, dtype=np.float32)
        ramp[start:] = t ** 1.55            # 아래로 갈수록 가파르게
    alpha = (ramp * strength).astype(np.uint8)
    a = np.repeat(alpha[:, None], w, axis=1)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    layer.putalpha(Image.fromarray(a, mode="L"))
    return layer


def text_overlay(size: tuple[int, int], lines: list[str], safe: tuple[float, float],
                 brand: str = BRAND, scrim: bool = True) -> Image.Image:
    """
    투명 RGBA 오버레이 (스크림 + 좌측정렬 하단 텍스트 + 브랜드).
    정지 소재와 영상 자막이 완전히 같은 경로를 쓴다.
    """
    w, h = size
    layer = bottom_scrim(size) if scrim else Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    x = int(w * SIDE_MARGIN)
    max_w = w - 2 * x
    font = fit_font(draw, lines, max_w, int(w * HEADLINE_SCALE))
    bfont = _font(max(12, int(w * BRAND_SCALE)))

    asc, desc = font.getmetrics()
    line_h = int((asc + desc) * LINE_SPACING)
    brand_h = sum(bfont.getmetrics())
    gap = int(font.size * BRAND_GAP)

    block_h = line_h * len(lines) + gap + brand_h
    bottom_safe = int(h * safe[1])
    top_safe = int(h * safe[0])

    y = h - bottom_safe - block_h
    y = max(top_safe, y)                    # 안전 영역 침범 방지

    for ln in lines:
        _draw_highlighted(draw, (x, y), ln, font)
        y += line_h

    y += gap
    draw.text((x, y), brand, font=bfont, fill=(235, 235, 235))
    return layer


def compose_still(photo: Image.Image, lines: list[str], ratio: str, anchor=None) -> Image.Image:
    spec = RATIOS[ratio]
    w, h = spec["size"]
    base = crop_to(photo, w, h, anchor).convert("RGBA")
    base.alpha_composite(text_overlay((w, h), lines, spec["safe"]))
    return base.convert("RGB")


# ── ffmpeg 헬퍼 ───────────────────────────────────────────────────────────
def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def probe(path) -> dict:
    out = run([FFPROBE, "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(path)]).stdout
    return json.loads(out)


def video_info(path) -> dict:
    d = probe(path)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    if not v:
        return {}
    # 세로 촬영본은 rotate 메타로 표현되기도 한다 → 표시 기준 해상도로 정규화
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(abs(float(sd["rotation"]))) % 180
    w, h = int(v["width"]), int(v["height"])
    if rot == 90:
        w, h = h, w
    fr = v.get("avg_frame_rate", "0/1")
    num, _, den = fr.partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return dict(w=w, h=h, fps=round(fps, 3),
                dur=float(d["format"].get("duration", 0.0)),
                rotation=rot, codec=v.get("codec_name", ""))


def rel(p) -> str:
    """표시용 경로. 출력 폴더가 저장소 밖이어도 죽지 않는다."""
    p = Path(p)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def load_cuts(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
