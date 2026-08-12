#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4단계 — 검수. 내보내기 전에 자동으로 잡을 수 있는 것은 전부 잡는다.

  python3 scripts/check.py [--stills 소재/stills] [--video 소재/video]

기계가 판정하는 항목
  · 규격(1080×1080 / 1080×1350 / 1080×1920)과 JPEG 여부
  · 한글 글리프 누락(두부 현상) — 폰트 cmap 대조
  · 텍스트가 좌우 여백(8.5%)·안전 영역을 침범하지 않는가
  · 사진이 90도 누워있지 않은가 (세로 소재가 가로로 나오지 않았는가)
  · 영상 코덱/픽셀포맷/fps/길이/무음/faststart

눈으로 봐야 하는 항목은 마지막에 목록으로 남긴다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BEATS, FFPROBE, FONT_PATH, HIGHLIGHT, RATIOS, REPO,  # noqa: E402
                    SIDE_MARGIN, BRAND, fit_font, rel, _font)

OK, BAD, WARN = "✅", "❌", "⚠️"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.failed = 0

    def add(self, mark: str, item: str, detail: str = "") -> None:
        self.rows.append((mark, item, detail))
        if mark == BAD:
            self.failed += 1
        print(f"{mark} {item}" + (f" — {detail}" if detail else ""))


def missing_glyphs() -> list[str]:
    """확정 카피 + 브랜드명의 모든 글자가 폰트에 있는지 확인한다 (두부 방지)."""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return []
    f = TTFont(str(FONT_PATH), fontNumber=0)
    cmap: set[int] = set()
    for t in f["cmap"].tables:
        cmap |= set(t.cmap.keys())
    text = BRAND + HIGHLIGHT + "".join("".join(b["lines"]) for b in BEATS)
    return sorted({c for c in text if c.strip() and ord(c) not in cmap})


def check_text_geometry(rep: Report) -> None:
    """실제 렌더링에 쓰는 계산으로 텍스트 박스를 재서 여백/안전영역 침범을 본다."""
    probe = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(probe)
    for ratio, spec in RATIOS.items():
        w, h = spec["size"]
        top_safe, bot_safe = int(h * spec["safe"][0]), int(h * spec["safe"][1])
        x = int(w * SIDE_MARGIN)
        max_w = w - 2 * x
        for b in BEATS:
            font = fit_font(d, b["lines"], max_w, int(w * 0.082))
            widest = max(int(d.textlength(ln, font=font)) for ln in b["lines"])
            if widest > max_w:
                rep.add(BAD, f"{b['id']} {ratio} 텍스트 폭",
                        f"{widest}px > 허용 {max_w}px (좌우 8.5% 여백 침범)")
                continue

            asc, desc = font.getmetrics()
            line_h = int((asc + desc) * 1.24)
            bfont = _font(max(12, int(w * 0.030)))
            block = line_h * len(b["lines"]) + int(font.size * 0.55) + sum(bfont.getmetrics())
            top = h - bot_safe - block
            if top < top_safe:
                rep.add(BAD, f"{b['id']} {ratio} 안전 영역",
                        f"블록 상단 {top}px < 상단 안전 {top_safe}px")
            else:
                rep.add(OK, f"{b['id']} {ratio}",
                        f"폭 {widest}/{max_w}px, 상단 여유 {top - top_safe}px, "
                        f"폰트 {font.size}px")


def check_stills(rep: Report, folder: Path, expect: str = "all") -> None:
    if not folder.is_dir():
        rep.add(WARN, "정지 소재", f"폴더 없음 {rel(folder)} — 아직 생성 전")
        return
    if expect == "all":
        expected = {f"{b['id']}_{r}.jpg" for b in BEATS for r in RATIOS}
    else:
        expected = {f"{n.strip()}.jpg" for n in expect.split(",") if n.strip()}
    found = {p.name for p in folder.glob("*.jpg")}
    for miss in sorted(expected - found):
        rep.add(BAD, "정지 소재 누락", miss)
    for name in sorted(found & expected):
        p = folder / name
        ratio = name.rsplit("_", 1)[1].removesuffix(".jpg")
        want = RATIOS[ratio]["size"]
        im = Image.open(p)
        if im.size != want:
            rep.add(BAD, f"{name} 규격", f"{im.size} ≠ {want}")
        elif im.format != "JPEG":
            rep.add(BAD, f"{name} 형식", im.format or "?")
        elif im.size[1] < im.size[0] and ratio != "1x1":
            rep.add(BAD, f"{name} 방향", "세로 소재가 가로로 누웠다")
        else:
            rep.add(OK, name, f"{im.size[0]}×{im.size[1]} JPEG {p.stat().st_size // 1024}KB")


def probe_json(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def check_video(rep: Report, folder: Path) -> None:
    if not folder.is_dir():
        rep.add(WARN, "영상 소재", f"폴더 없음 {rel(folder)} — 아직 생성 전")
        return
    for ratio in ("9x16", "1x1"):
        p = folder / f"ad_{ratio}.mp4"
        if not p.exists():
            rep.add(BAD, "영상 누락", p.name)
            continue
        d = probe_json(p)
        v = next((s for s in d["streams"] if s["codec_type"] == "video"), {})
        audio = [s for s in d["streams"] if s["codec_type"] == "audio"]
        want_w, want_h = RATIOS[ratio]["size"]
        dur = float(d["format"].get("duration", 0))
        fr = v.get("avg_frame_rate", "0/1")
        num, _, den = fr.partition("/")
        fps = float(num) / float(den or 1)

        issues = []
        if (v.get("width"), v.get("height")) != (want_w, want_h):
            issues.append(f"규격 {v.get('width')}×{v.get('height')} ≠ {want_w}×{want_h}")
        if v.get("codec_name") != "h264":
            issues.append(f"코덱 {v.get('codec_name')}")
        if v.get("pix_fmt") != "yuv420p":
            issues.append(f"픽셀포맷 {v.get('pix_fmt')}")
        if abs(fps - 30) > 0.1:
            issues.append(f"fps {fps}")
        if abs(dur - 12.0) > 0.15:
            issues.append(f"길이 {dur:.2f}s ≠ 12s")
        if audio:
            issues.append("오디오 트랙 있음 (무음이어야 함)")
        if b"moov" not in p.open("rb").read(4096):
            issues.append("faststart 안 걸림 (moov 뒤쪽)")

        if issues:
            rep.add(BAD, p.name, "; ".join(issues))
        else:
            rep.add(OK, p.name,
                    f"{want_w}×{want_h} h264/yuv420p {fps:.0f}fps {dur:.2f}s 무음 faststart")


EYE = [
    "책 제목이 크롭에서 잘리지 않았는가 (표지 글자 끝까지 보이는가)",
    "영상 각 컷이 흔들리지 않는가 (실제로 재생해서 확인)",
    "비트 순서가 문제 → 전환 → 해결로 흐르는가",
    "시선이 비트를 따라 이동하는가 (단권 → 다른 책/내지 → 두 권 함께)",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", default=str(REPO / "소재" / "stills"))
    ap.add_argument("--video", default=str(REPO / "소재" / "video"))
    ap.add_argument("--expect", default="all",
                    help='기대하는 정지 소재. "all"(비트3×비율3) 또는 "b1_4x5,b3_4x5" 처럼 지정')
    a = ap.parse_args()

    rep = Report()

    miss = missing_glyphs()
    if miss:
        rep.add(BAD, "한글 글리프", f"폰트에 없는 글자: {''.join(miss)}")
    else:
        rep.add(OK, "한글 글리프", f"확정 카피 전체 렌더 가능 ({FONT_PATH.name})")

    print("\n── 텍스트 배치 ──")
    check_text_geometry(rep)
    print("\n── 정지 소재 ──")
    check_stills(rep, Path(a.stills), a.expect)
    print("\n── 영상 소재 ──")
    check_video(rep, Path(a.video))

    print("\n── 눈으로 확인할 항목 ──")
    for e in EYE:
        print(f"[ ] {e}")

    print(f"\n실패 {rep.failed}건")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
