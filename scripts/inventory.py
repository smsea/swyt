#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0단계 — 인벤토리.

원본 폴더의 사진/영상을 전부 열어 측정하고, 눈으로 확인할 컨택트시트를 굽는다.
파일명이 타임스탬프뿐이라 아무 정보가 없으므로, 기계로 잴 수 있는 것은 전부 재고
(방향/해상도/초점/노출/여백), 사람 눈이 필요한 칸(어느 책·구도)은 비워서 채우게 한다.

  python3 scripts/inventory.py --src <원본폴더> [--out 소재] [--workers 4]

영상은 프레임을 훑어 흔들림이 적고 책이 또렷한 구간을 초 단위로 찾아 기록한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FFMPEG, REPO, load_image, video_info  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
VID_EXT = {".mp4", ".mov", ".m4v", ".avi"}

THUMB = 420          # 컨택트시트 셀 크기
SAMPLE_FPS = 2.0     # 영상 분석 샘플링
MIN_WINDOW = 4.0     # 비트당 4초 → 최소 확보 구간


# ── 측정 ──────────────────────────────────────────────────────────────────
def sharpness(gray: np.ndarray) -> float:
    """라플라시안 분산. 값이 낮을수록 흐리거나 흔들린 것."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    g = gray.astype(np.float32)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    win = np.lib.stride_tricks.sliding_window_view(g, (3, 3))
    lap = (win * k).sum(axis=(-2, -1))
    return float(lap.var())


def _gray(img: Image.Image, box: int = 512) -> np.ndarray:
    im = img.convert("L")
    im.thumbnail((box, box), Image.BILINEAR)
    return np.asarray(im)


def empty_space(img: Image.Image) -> dict:
    """
    텍스트를 얹을 빈 공간이 어디인가.
    4분면(상/하/좌/우) 밴드별 디테일(엣지 밀도)과 밝기 분산을 재서
    값이 낮은 쪽 = 비어있는 쪽으로 본다.
    """
    g = _gray(img, 256).astype(np.float32)
    h, w = g.shape
    gy, gx = np.gradient(g)
    det = np.hypot(gx, gy)
    bands = {
        "상": det[: int(h * 0.3), :],
        "하": det[int(h * 0.7):, :],
        "좌": det[:, : int(w * 0.3)],
        "우": det[:, int(w * 0.7):],
    }
    scores = {k: float(v.mean()) for k, v in bands.items()}
    lo = min(scores.values()) or 1.0
    # 가장 조용한 밴드부터 나열 (여백 후보)
    order = sorted(scores, key=lambda k: scores[k])
    return dict(scores={k: round(v, 1) for k, v in scores.items()},
                free=order[:2], quietest=order[0], ratio=round(max(scores.values()) / lo, 2))


def exposure(img: Image.Image) -> dict:
    g = _gray(img, 256).astype(np.float32)
    mean = float(g.mean())
    clip_hi = float((g > 250).mean() * 100)
    clip_lo = float((g < 5).mean() * 100)
    if mean < 70:
        verdict = "어두움"
    elif mean > 190:
        verdict = "밝음"
    else:
        verdict = "적정"
    return dict(mean=round(mean, 1), blown=round(clip_hi, 2),
                crushed=round(clip_lo, 2), verdict=verdict)


# ── 사진 ──────────────────────────────────────────────────────────────────
def analyse_photo(path: Path, thumbs: Path) -> dict:
    img = load_image(path)            # exif_transpose 적용됨
    w, h = img.size
    sharp = sharpness(_gray(img))
    exp = exposure(img)
    free = empty_space(img)

    t = img.copy()
    t.thumbnail((THUMB, THUMB), Image.LANCZOS)
    tp = thumbs / f"{path.stem}.jpg"
    t.save(tp, "JPEG", quality=88)

    return dict(
        kind="photo", file=path.name, path=str(path), thumb=str(tp),
        w=w, h=h, orient="세로" if h >= w else "가로",
        mp=round(w * h / 1e6, 1),
        sharp=round(sharp, 1),
        focus="양호" if sharp > 90 else ("주의" if sharp > 40 else "흐림"),
        exposure=exp["verdict"], exp_mean=exp["mean"],
        blown=exp["blown"], crushed=exp["crushed"],
        free=" ".join(free["free"]), free_scores=free["scores"],
    )


# ── 영상 ──────────────────────────────────────────────────────────────────
def sample_frames(path: Path, tmp: Path, fps: float = SAMPLE_FPS) -> list[np.ndarray]:
    tmp.mkdir(parents=True, exist_ok=True)
    for old in tmp.glob("*.jpg"):
        old.unlink()
    subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale=320:-2",
         "-q:v", "4", str(tmp / "f_%05d.jpg")],
        check=True, capture_output=True,
    )
    out = []
    for f in sorted(tmp.glob("f_*.jpg")):
        out.append(np.asarray(Image.open(f).convert("L"), dtype=np.float32))
    return out


def best_window(frames: list[np.ndarray], fps: float, need: float = MIN_WINDOW) -> dict:
    """
    흔들림이 적고 책이 또렷하게 잡히는 구간을 초 단위로 찾는다.
    프레임별 선명도와 인접 프레임 간 움직임을 합쳐 점수화한 뒤,
    need 초 길이의 슬라이딩 윈도우 중 최고 점수 구간을 고른다.
    """
    n = len(frames)
    if n < 2:
        return {}
    sharp = np.array([sharpness(f) for f in frames], dtype=np.float32)
    motion = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        motion[i] = float(np.abs(frames[i] - frames[i - 1]).mean())
    motion[0] = motion[1]

    s_n = sharp / (sharp.max() or 1.0)
    m_n = motion / (motion.max() or 1.0)
    score = s_n - 1.15 * m_n                  # 선명하고 덜 흔들릴수록 높다

    win = max(1, int(round(need * fps)))
    if n <= win:
        return dict(start=0.0, dur=round(n / fps, 2),
                    score=round(float(score.mean()), 3),
                    sharp=round(float(sharp.mean()), 1),
                    motion=round(float(motion.mean()), 2))

    csum = np.concatenate([[0.0], np.cumsum(score)])
    means = (csum[win:] - csum[:-win]) / win
    i = int(np.argmax(means))
    return dict(
        start=round(i / fps, 2), dur=round(win / fps, 2),
        score=round(float(means[i]), 3),
        sharp=round(float(sharp[i:i + win].mean()), 1),
        motion=round(float(motion[i:i + win].mean()), 2),
        steady="양호" if float(motion[i:i + win].mean()) < 6 else "흔들림 주의",
    )


def analyse_video(path: Path, thumbs: Path, tmp: Path) -> dict:
    info = video_info(path)
    frames = sample_frames(path, tmp)
    win = best_window(frames, SAMPLE_FPS)

    # 첫 프레임 + 좋은 구간 중간 프레임을 뽑아 눈으로 확인할 수 있게 한다
    stamps = [0.0, round(win.get("start", 0.0) + win.get("dur", 0.0) / 2, 2)]
    shots = []
    for i, ts in enumerate(stamps):
        op = thumbs / f"{path.stem}_t{i}.jpg"
        subprocess.run(
            [FFMPEG, "-v", "error", "-ss", str(ts), "-i", str(path),
             "-frames:v", "1", "-vf", f"scale={THUMB}:-2", "-q:v", "3", "-y", str(op)],
            check=True, capture_output=True,
        )
        if op.exists():
            shots.append(str(op))

    return dict(
        kind="video", file=path.name, path=str(path),
        thumb=shots[0] if shots else "", thumbs=shots,
        w=info.get("w", 0), h=info.get("h", 0),
        orient="세로" if info.get("h", 0) >= info.get("w", 1) else "가로",
        dur=round(info.get("dur", 0.0), 2), fps=info.get("fps", 0),
        rotation=info.get("rotation", 0),
        good_start=win.get("start"), good_dur=win.get("dur"),
        steady=win.get("steady", ""), sharp=win.get("sharp"),
        motion=win.get("motion"), score=win.get("score"),
    )


# ── 컨택트시트 ────────────────────────────────────────────────────────────
def contact_sheet(rows: list[dict], out: Path, cols: int = 5, cell: int = THUMB) -> Path:
    items = [r for r in rows if r.get("thumb")]
    if not items:
        return out
    n = len(items)
    rows_n = (n + cols - 1) // cols
    label = 34
    sheet = Image.new("RGB", (cols * cell, rows_n * (cell + label)), (24, 24, 26))
    from PIL import ImageDraw, ImageFont
    from common import FONT_PATH
    f = ImageFont.truetype(str(FONT_PATH), 17)
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(items):
        cx, cy = (i % cols) * cell, (i // cols) * (cell + label)
        try:
            th = Image.open(REPO / r["thumb"]).convert("RGB")
        except Exception:
            continue
        th.thumbnail((cell, cell), Image.LANCZOS)
        sheet.paste(th, (cx + (cell - th.width) // 2, cy + (cell - th.height) // 2))
        tag = r["file"]
        if r["kind"] == "video" and r.get("good_start") is not None:
            tag += f"  ▶{r['good_start']}s+{r['good_dur']}s"
        d.text((cx + 6, cy + cell + 7), tag[:40], font=f, fill=(225, 225, 225))
    sheet.save(out, "JPEG", quality=88)
    return out


# ── 리포트 ────────────────────────────────────────────────────────────────
PHOTO_HDR = ["파일", "어느 책", "구도", "세로/가로", "해상도", "초점", "노출",
             "여백", "비고"]
VIDEO_HDR = ["파일", "어느 책", "구도", "세로/가로", "해상도", "길이", "fps",
             "좋은 구간", "흔들림", "비고"]


def write_markdown(photos: list[dict], videos: list[dict], out: Path, src: Path) -> None:
    L = []
    L.append("# 인벤토리 — 전자책 광고 소재 원본\n")
    L.append(f"- 원본 폴더: `{src}`")
    L.append(f"- 사진 {len(photos)}장 / 영상 {len(videos)}건")
    L.append("- 「어느 책」·「구도」 칸은 컨택트시트를 눈으로 보고 `labels.json` 에 적는다.")
    L.append("  - 어느 책: 검색되는 블로그 글쓰기법 / 예약을 부르는 블로그 운영법 / 두 권 함께 / 내지")
    L.append("  - 구도: 표지 정면 / 비스듬 / 손에 든 컷 / 책상 연출 / 펼친 내지 / 여러 권 쌓기")
    L.append("- 나머지 칸은 스크립트가 측정한 값이다.\n")
    L.append("컨택트시트: `소재/contact_photos.jpg`, `소재/contact_videos.jpg`\n")

    L.append("## 사진\n")
    L.append("| " + " | ".join(PHOTO_HDR) + " |")
    L.append("|" + "---|" * len(PHOTO_HDR))
    for r in photos:
        L.append("| {f} | {bk} | {sh} | {o} | {w}×{h} | {fo} ({s}) | {e} | {fr} | {n} |".format(
            f=r["file"], bk=r.get("book", ""), sh=r.get("shot", ""),
            o=r["orient"], w=r["w"], h=r["h"],
            fo=r["focus"], s=r["sharp"], e=r["exposure"], fr=r["free"],
            n=r.get("note", "")))

    L.append("\n## 영상\n")
    L.append("| " + " | ".join(VIDEO_HDR) + " |")
    L.append("|" + "---|" * len(VIDEO_HDR))
    for r in videos:
        gs = "-" if r.get("good_start") is None else f"{r['good_start']}s ~ +{r['good_dur']}s"
        L.append("| {f} | {bk} | {sh} | {o} | {w}×{h} | {d}s | {fp} | {g} | {st} | {n} |".format(
            f=r["file"], bk=r.get("book", ""), sh=r.get("shot", ""),
            o=r["orient"], w=r["w"], h=r["h"],
            d=r["dur"], fp=r["fps"], g=gs, st=r.get("steady", ""),
            n=r.get("note", "")))

    L.append("\n## 품질 상세 (측정값)\n")
    L.append("| 파일 | 선명도 | 노출 평균 | 날림% | 뭉갬% | 여백 점수 |")
    L.append("|---|---|---|---|---|---|")
    for r in photos:
        L.append("| {f} | {s} | {m} | {b} | {c} | {fs} |".format(
            f=r["file"], s=r["sharp"], m=r["exp_mean"], b=r["blown"],
            c=r["crushed"], fs=json.dumps(r["free_scores"], ensure_ascii=False)))

    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 사진·영상 폴더")
    ap.add_argument("--out", default=str(REPO / "소재"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--labels", default=str(REPO / "labels.json"),
                    help="눈으로 판정한 어느 책/구도 분류. 표에 병합한다.")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if not src.is_dir():
        print(f"원본 폴더가 없다: {src}", file=sys.stderr)
        return 2

    thumbs = out / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    tmp = out / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.rglob("*") if p.is_file())
    pics = [p for p in files if p.suffix.lower() in IMG_EXT]
    vids = [p for p in files if p.suffix.lower() in VID_EXT]
    print(f"사진 {len(pics)}장, 영상 {len(vids)}건 확인")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        photos = list(ex.map(lambda p: analyse_photo(p, thumbs), pics))
        videos = list(ex.map(lambda p: analyse_video(p, thumbs, tmp / p.stem), vids))

    photos.sort(key=lambda r: r["file"])
    videos.sort(key=lambda r: r["file"])

    labels = {}
    lp = Path(a.labels)
    if lp.exists():
        with open(lp, encoding="utf-8") as f:
            labels = json.load(f).get("labels", {})
        print(f"분류 {len(labels)}건 병합: {lp.name}")
    for r in photos + videos:
        lab = labels.get(r["file"], {})
        r["book"] = lab.get("book", "")
        r["shot"] = lab.get("shot", "")
        r["note"] = lab.get("note", "")

    contact_sheet(photos, out / "contact_photos.jpg")
    contact_sheet(videos, out / "contact_videos.jpg")
    write_markdown(photos, videos, out / "inventory.md", src)

    with open(out / "inventory.json", "w", encoding="utf-8") as f:
        json.dump(dict(src=str(src), photos=photos, videos=videos), f,
                  ensure_ascii=False, indent=2)

    with open(out / "inventory.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "file", "orient", "w", "h", "focus_or_steady",
                    "exposure_or_dur", "free_or_window"])
        for r in photos:
            w.writerow(["photo", r["file"], r["orient"], r["w"], r["h"],
                        r["focus"], r["exposure"], r["free"]])
        for r in videos:
            w.writerow(["video", r["file"], r["orient"], r["w"], r["h"],
                        r.get("steady", ""), r["dur"],
                        f"{r.get('good_start')}+{r.get('good_dur')}"])

    print(f"→ {out/'inventory.md'}")
    print(f"→ {out/'contact_photos.jpg'} / {out/'contact_videos.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
