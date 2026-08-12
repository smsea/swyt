#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2단계 — 정지 소재. 비트 3개 × 비율 3종 = 9장.

  python3 scripts/make_stills.py --cuts cuts.json [--out 소재/stills] [--beat b1]

컷 교체는 cuts.json 의 photo / anchor 만 바꾸면 된다. 코드는 건드리지 않는다.
한글은 전부 PIL 로 렌더링해서 굽는다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BEATS, JPEG_QUALITY, RATIOS, REPO, compose_still,  # noqa: E402
                    load_cuts, load_image, rel)
from compose import compose_designed  # noqa: E402


def resolve_beats(cuts: dict) -> list[dict]:
    """cuts.json 의 비트 설정을 확정 카피(BEATS)와 합친다. 카피는 덮어쓰지 않는다."""
    by_id = {b["id"]: b for b in cuts.get("beats", [])}
    out = []
    for b in BEATS:
        c = by_id.get(b["id"], {})
        out.append(dict(
            id=b["id"], role=b["role"], lines=b["lines"],
            photo=c.get("photo"),
            bg_photo=c.get("bg_photo"),
            subject_scale=float(c.get("subject_scale", 1.0)),
            anchor=tuple(c["anchor"]) if c.get("anchor") else None,
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", default=str(REPO / "cuts.json"))
    ap.add_argument("--out", default=str(REPO / "소재" / "stills"))
    ap.add_argument("--beat", action="append", help="특정 비트만 (b1/b2/b3), 반복 가능")
    ap.add_argument("--ratio", action="append", help="특정 비율만 (1x1/4x5/9x16), 반복 가능")
    ap.add_argument("--style", choices=("designed", "crop"), default="designed",
                    help="designed=배경 재구성 후 합성(기본) / crop=원본 크롭 위에 텍스트")
    ap.add_argument("--cache", default=str(REPO / "소재" / ".cache"))
    a = ap.parse_args()

    cuts = load_cuts(a.cuts)
    beats = resolve_beats(cuts)
    if a.beat:
        beats = [b for b in beats if b["id"] in a.beat]
    ratios = a.ratio or list(RATIOS)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    made, missing = [], []
    for b in beats:
        if not b["photo"]:
            missing.append(f"{b['id']}: cuts.json 에 photo 가 없다")
            continue
        p = Path(b["photo"])
        if not p.is_absolute():
            p = REPO / p
        if not p.exists():
            missing.append(f"{b['id']}: 사진 없음 {p}")
            continue

        bg = Path(b["bg_photo"]) if b.get("bg_photo") else None
        if bg is not None and not bg.is_absolute():
            bg = REPO / bg
        photo = load_image(p) if a.style == "crop" else None   # exif_transpose 적용

        for r in ratios:
            if a.style == "designed":
                img = compose_designed(p, b["lines"], r, Path(a.cache), bg_photo=bg,
                                       subject_scale=b.get("subject_scale", 1.0))
            else:
                img = compose_still(photo, b["lines"], r, anchor=b["anchor"])
            fp = out / f"{b['id']}_{r}.jpg"
            img.save(fp, "JPEG", quality=JPEG_QUALITY, subsampling=1, optimize=True)
            made.append(fp)
            print(f"{rel(fp)}  {img.size[0]}×{img.size[1]}  ({b['role']}) ← {p.name}")

    for m in missing:
        print(f"[건너뜀] {m}", file=sys.stderr)
    print(f"\n{len(made)}장 생성")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
