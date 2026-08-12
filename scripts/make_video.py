#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3단계 — 영상 소재. 9:16 / 1:1 두 벌, 각 12초(비트당 4초).

  python3 scripts/make_video.py --cuts cuts.json [--out 소재/video] [--ratio 9x16]

· 0단계에서 기록한 좋은 구간에서 잘라 쓴다 (cuts.json 의 video.start).
· 비트별 세그먼트를 각각 만든 뒤 concat 데믹서로 조립한다. 자동 편집 필터는 쓰지 않는다.
· 자막은 ffmpeg drawtext 가 아니라 PIL 로 만든 투명 PNG 를 overlay 로 합성한다.
· 무음(-an). H.264 / yuv420p / 30fps / CRF 20 / +faststart.
· 쓸 만한 영상 구간이 없는 비트는 사진에 Ken Burns 줌(1.12배)으로 대체한다.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BEATS, FFMPEG, RATIOS, REPO, load_cuts, rel, text_overlay)  # noqa: E402
from compose import compose_designed  # noqa: E402

VIDEO_RATIOS = ["9x16", "1x1"]
BEAT_SEC = 4.0
FPS = 30
CRF = 20
KEN_BURNS = 1.12

ENC = ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
       "-preset", "medium", "-crf", str(CRF), "-r", str(FPS),
       "-x264-params", "keyint=60:min-keyint=60:scenecut=0", "-an"]


def _resolve(p: str | None) -> Path | None:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else (REPO / q)


def cover_crop(w: int, h: int, fx: float, fy: float) -> str:
    """대상 비율로 꽉 채운 뒤 기준점에 맞춰 잘라낸다. 단순 중앙 크롭이 아니다."""
    return (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(iw-ow)*{fx:.4f}:(ih-oh)*{fy:.4f}")


def segment_from_video(src: Path, start: float, dur: float, overlay: Path,
                       w: int, h: int, fx: float, fy: float, out: Path) -> None:
    vf = f"{cover_crop(w, h, fx, fy)},fps={FPS},setsar=1"
    cmd = [FFMPEG, "-v", "error", "-y",
           "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
           "-i", str(overlay),
           "-filter_complex", f"[0:v]{vf}[v];[v][1:v]overlay=0:0:format=auto[o]",
           "-map", "[o]", *ENC, str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def segment_from_photo(src: Path, dur: float, overlay: Path,
                       w: int, h: int, fx: float, fy: float, out: Path) -> None:
    """사진 대체분 — Ken Burns 줌 1.12배."""
    frames = int(round(dur * FPS))
    big_w, big_h = w * 2, h * 2
    z = f"min(1+{KEN_BURNS - 1:.4f}*on/{max(frames - 1, 1)},{KEN_BURNS})"
    vf = (f"{cover_crop(w, h, fx, fy)},scale={big_w}:{big_h},"
          f"zoompan=z='{z}':d={frames}"
          f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
          f":s={w}x{h}:fps={FPS},setsar=1")
    cmd = [FFMPEG, "-v", "error", "-y",
           "-loop", "1", "-t", f"{dur:.3f}", "-i", str(src),
           "-i", str(overlay),
           "-filter_complex", f"[0:v]{vf}[v];[v][1:v]overlay=0:0:format=auto[o]",
           "-map", "[o]", "-frames:v", str(frames), *ENC, str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def concat(segments: list[Path], out: Path, work: Path) -> None:
    """concat 데믹서로 조립한다."""
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in segments), encoding="utf-8")
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", "-movflags", "+faststart", str(out)],
        check=True, capture_output=True, text=True)


def build(ratio: str, beats: list[dict], out_dir: Path, work: Path,
          style: str = "designed", cache: Path | None = None) -> Path | None:
    w, h = RATIOS[ratio]["size"]
    safe = RATIOS[ratio]["safe"]
    segs = []

    for b in beats:
        ov = work / f"ov_{ratio}_{b['id']}.png"
        text_overlay((w, h), b["lines"], safe).save(ov)

        seg = work / f"seg_{ratio}_{b['id']}.mp4"
        fx, fy = b["anchor_v"]
        vid, photo = _resolve(b.get("video_file")), _resolve(b.get("photo"))

        if style == "designed" and photo and photo.exists():
            # 정지 소재와 같은 합성 화면(자막 제외)을 만들어 Ken Burns 를 건다.
            base = work / f"base_{ratio}_{b['id']}.png"
            compose_designed(photo, b["lines"], ratio, cache,
                             bg_photo=_resolve(b.get("bg_photo")),
                             with_text=False, scale=2,
                             subject_scale=b.get("subject_scale", 1.0)).save(base)
            segment_from_photo(base, BEAT_SEC, ov, w, h, fx, fy, seg)
            print(f"  {b['id']} ← {photo.name} 합성 화면 + Ken Burns {KEN_BURNS}배")
        elif vid and vid.exists():
            segment_from_video(vid, b["video_start"], BEAT_SEC, ov, w, h, fx, fy, seg)
            print(f"  {b['id']} ← {vid.name} @{b['video_start']}s +{BEAT_SEC}s")
        elif photo and photo.exists():
            segment_from_photo(photo, BEAT_SEC, ov, w, h, fx, fy, seg)
            print(f"  {b['id']} ← {photo.name} (Ken Burns {KEN_BURNS}배 대체)")
        else:
            print(f"  [건너뜀] {b['id']}: 쓸 영상도 사진도 없다", file=sys.stderr)
            return None
        segs.append(seg)

    out = out_dir / f"ad_{ratio}.mp4"
    concat(segs, out, work)
    return out


def resolve_beats(cuts: dict) -> list[dict]:
    by_id = {b["id"]: b for b in cuts.get("beats", [])}
    out = []
    for b in BEATS:
        c = by_id.get(b["id"], {})
        v = c.get("video") or {}
        out.append(dict(
            id=b["id"], role=b["role"], lines=b["lines"],
            photo=c.get("photo"),
            bg_photo=c.get("bg_photo"),
            subject_scale=float(c.get("subject_scale", 1.0)),
            video_file=v.get("file"),
            video_start=float(v.get("start", 0.0)),
            anchor_v=tuple(c.get("anchor_video", c.get("anchor_v", (0.5, 0.40)))),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", default=str(REPO / "cuts.json"))
    ap.add_argument("--out", default=str(REPO / "소재" / "video"))
    ap.add_argument("--ratio", action="append", help="9x16 / 1x1, 반복 가능")
    ap.add_argument("--keep-work", action="store_true", help="세그먼트 중간물 보존")
    ap.add_argument("--style", choices=("designed", "raw"), default="designed",
                    help="designed=합성 화면에 Ken Burns(기본) / raw=원본 영상 구간 사용")
    ap.add_argument("--cache", default=str(REPO / "소재" / ".cache"))
    a = ap.parse_args()

    cuts = load_cuts(a.cuts)
    beats = resolve_beats(cuts)
    ratios = a.ratio or VIDEO_RATIOS

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="swyt_seg_"))

    made = []
    try:
        for r in ratios:
            print(f"[{r}]")
            f = build(r, beats, out_dir, work, style=a.style, cache=Path(a.cache))
            if f:
                made.append(f)
                print(f"→ {rel(f)}")
    finally:
        if a.keep_work:
            print(f"세그먼트: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    print(f"\n{len(made)}개 생성")
    return 0 if len(made) == len(ratios) else 1


if __name__ == "__main__":
    raise SystemExit(main())
