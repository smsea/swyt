# -*- coding: utf-8 -*-
"""
배경 재구성 합성 — 사진을 그대로 붙이지 않는다.

원본에서 책(+든 손)을 분리해 내고, 같은 사진의 배경을 크게 흐려 색조를 정리한 뒤
그 위에 다시 앉힌다. 배경 색이 원본에서 나오므로 책과 따로 놀지 않고,
흐림·감광·비네트를 거치면서 스냅사진이 아니라 설계된 화면이 된다.

분리 결과는 소재/.cache 에 캐시한다 (한 장에 수 초 걸린다).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from common import RATIOS, crop_to, load_image, text_block_top, text_overlay

# 비트 텍스트가 앉을 자리를 비워둔, 피사체가 들어갈 상자 (좌, 상, 우, 하 — 캔버스 비율)
SUBJECT_BOX = {
    "1x1":  (0.10, 0.07, 0.90, 0.58),
    "4x5":  (0.11, 0.07, 0.89, 0.64),
    "9x16": (0.11, 0.15, 0.89, 0.62),
}


# 책이 캔버스 높이의 몇 %를 차지할지 — 기울기·소품과 무관하게 컷마다 같은 크기로 보이게
BOOK_HEIGHT = {"1x1": 0.42, "4x5": 0.50, "9x16": 0.46}
MAX_WIDTH = 0.82         # 비스듬한 컷이 옆으로 넘치지 않게 하는 폭 상한

BLUR_FRAC = 0.030        # 배경 흐림 (캔버스 너비 대비)
BG_SAT = 0.90            # 배경 채도 — 장소색은 남기되 책이 주인공이 되게
BG_BRIGHT = 0.74         # 배경 감광 — 컷별 밝기 차이가 살아있어야 한다
VIGNETTE = 0.30          # 가장자리 어둡기
SHADOW_BLUR = 0.022      # 그림자 번짐
SHADOW_ALPHA = 130
SHADOW_DY = 0.018        # 그림자 아래로 밀기
BOOK_TEXT_GAP = 0.025    # 책 아랫변과 헤드라인 사이 최소 간격 (캔버스 높이 대비)
MIN_TOP = 0.05           # 책 윗변이 캔버스 위로 붙을 수 있는 한계

_session = None


def _rembg_session():
    global _session
    if _session is None:
        from rembg import new_session
        _session = new_session("u2net")
    return _session


def cutout(path: Path, cache_dir: Path, max_side: int = 1800) -> Image.Image:
    """책 + 든 손을 배경에서 분리한다. 결과는 캐시한다."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"{Path(path).stem}_cut.png"
    if cp.exists():
        return Image.open(cp).convert("RGBA")

    from rembg import remove
    im = load_image(path)                      # exif_transpose 적용
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    out = remove(im, session=_rembg_session()).convert("RGBA")
    bbox = out.getbbox()
    if bbox:
        out = out.crop(bbox)                   # 여백 제거 — 배치 계산이 정확해진다
    out.save(cp)
    return out


def _vignette(size: tuple[int, int], strength: float) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2)
    a = np.clip(r ** 1.7, 0, 1) * strength
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    layer.putalpha(Image.fromarray((a * 255).astype(np.uint8), mode="L"))
    return layer


def make_background(src: Image.Image, w: int, h: int) -> Image.Image:
    """같은 사진의 배경을 크게 흐리고 눌러 무대로 쓴다. 색조가 책과 이어진다."""
    bg = crop_to(src, w, h)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=max(4, int(w * BLUR_FRAC))))
    bg = ImageEnhance.Color(bg).enhance(BG_SAT)
    bg = ImageEnhance.Brightness(bg).enhance(BG_BRIGHT)
    bg = bg.convert("RGBA")
    bg.alpha_composite(_vignette((w, h), VIGNETTE))
    return bg


def _shadow(sub: Image.Image, w: int, canvas: tuple[int, int], xy: tuple[int, int]) -> Image.Image:
    """피사체 알파에서 만든 부드러운 그림자. 책이 배경 위에 떠 있게 만든다."""
    layer = Image.new("RGBA", canvas, (0, 0, 0, 0))
    a = sub.split()[-1].point(lambda v: min(255, int(v * SHADOW_ALPHA / 255 * 1.6)))
    sh = Image.new("RGBA", sub.size, (0, 0, 0, 0))
    sh.putalpha(a)
    layer.paste(sh, (xy[0], xy[1] + int(canvas[1] * SHADOW_DY)), sh)
    return layer.filter(ImageFilter.GaussianBlur(radius=max(6, int(w * SHADOW_BLUR))))


def book_bottom_frac(sub: Image.Image) -> float:
    """
    잘라낸 영역에서 책이 끝나고 손목이 시작되는 지점을 찾는다 (0~1).

    책은 넓고 손목은 좁다. 위에서 내려오며 행 폭이 최대 폭의 절반 밑으로
    떨어지는 첫 지점이 책의 아랫변이다. 이 값이 있어야 '책이 헤드라인을
    덮지 않게' 크기를 자동으로 줄일 수 있다.
    """
    a = np.asarray(sub.split()[-1])
    widths = (a > 128).sum(axis=1)
    if widths.max() == 0:
        return 1.0
    thresh = widths.max() * 0.55
    below = np.where(widths < thresh)[0]
    # 맨 위 몇 줄은 원래 좁을 수 있으니 폭이 한 번 최대에 닿은 뒤부터 본다
    peak = int(np.argmax(widths))
    after = below[below > peak]
    return float(after[0] / len(widths)) if len(after) else 1.0


def place_subject(bg: Image.Image, sub: Image.Image, ratio: str,
                  text_top: int | None = None, subject_scale: float = 1.0) -> Image.Image:
    """
    책 아랫변을 헤드라인 바로 위 기준선에 맞춰 앉힌다.

    크기는 '책 높이'로 정한다. 폭 기준으로 잡으면 비스듬한 컷이나 소품·손이 함께
    잘린 컷에서 책만 작아져 비트마다 크기가 흔들린다.
    피사체는 비율 그대로 통째로 움직인다. 늘이거나 눌러 붙이지 않는다 —
    손목만 늘였더니 고무팔이 됐다. 화면 밖으로 나가는 부분은 그냥 잘린다.
    화병처럼 소품이 함께 잘려 높이 기준이 흔들리는 컷은 cuts.json 의
    subject_scale 로 미세 조정한다.
    """
    w, h = bg.size
    bf = book_bottom_frac(sub)
    limit = (text_top if text_top is not None else int(h * 0.62)) - int(h * BOOK_TEXT_GAP)
    min_top = int(h * MIN_TOP)

    # 책 '높이'를 기준으로 크기를 정한다. 폭 기준으로 잡으면 비스듬한 컷이나
    # 소품이 함께 잘린 컷에서 책만 작아진다 — 비트마다 책 크기가 흔들린다.
    book_h = BOOK_HEIGHT[ratio] * h
    book_h = min(book_h, limit - min_top)
    s = book_h * subject_scale / max(1.0, sub.height * bf)
    if sub.width * s > w * MAX_WIDTH:            # 비스듬한 컷이 옆으로 넘치지 않게
        s = w * MAX_WIDTH / sub.width

    nw, nh = max(1, int(sub.width * s)), max(1, int(sub.height * s))
    sub = sub.resize((nw, nh), Image.LANCZOS)

    # 책 아랫변을 헤드라인 바로 위 기준선에 맞춘다. 손·소품은 비율 그대로 따라
    # 내려가고, 화면 밖으로 나가는 부분은 자연스럽게 잘린다.
    x = int((w - nw) / 2)
    y = int(limit - nh * bf)
    y = max(min_top if nh * bf >= h * 0.2 else 0, y)
    out = bg.copy()
    out.alpha_composite(_shadow(sub, w, (w, h), (x, y)))
    out.alpha_composite(sub, (x, y))
    return out


def compose_designed(photo: Path, lines: list[str], ratio: str,
                     cache_dir: Path, bg_photo: Path | None = None,
                     with_text: bool = True, scale: int = 1,
                     subject_scale: float = 1.0) -> Image.Image:
    """
    배경 재구성 + 책 합성 (+ 스크림/텍스트).

    with_text=False 는 영상용. 자막은 매 프레임 얹지 않고, Ken Burns 로 움직이는
    화면 위에 ffmpeg overlay 로 한 번만 합성한다 (자막이 같이 확대되면 안 된다).
    scale 을 올리면 확대해도 뭉개지지 않는 큰 화면을 만든다.
    """
    spec = RATIOS[ratio]
    w, h = spec["size"][0] * scale, spec["size"][1] * scale

    src = load_image(bg_photo or photo)
    canvas = make_background(src, w, h)
    text_top = text_block_top((w, h), lines, spec["safe"])
    canvas = place_subject(canvas, cutout(photo, cache_dir), ratio, text_top=text_top,
                           subject_scale=subject_scale)
    if with_text:
        canvas.alpha_composite(text_overlay((w, h), lines, spec["safe"]))
    return canvas.convert("RGB")
