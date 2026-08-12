# 전자책 광고 소재 파이프라인 — 다미앤마케팅

Meta Ads 에 바로 올릴 소재를 만드는 재실행 가능한 스크립트.
쓸 사진·영상과 구간은 전부 `cuts.json` 인자로 바꾼다. 컷 교체는 코드를 건드리지 않는다.

## 현재 상태

**0단계(인벤토리)는 아직 끝나지 않았다.** 이 세션의 실행 환경에서 촬영 원본에 접근할
수 없기 때문이다 (아래 「원본 확보」 참고). 파이프라인 자체는 합성 원본으로 전 구간을
돌려 검증을 마쳤다 — 원본만 있으면 바로 인벤토리부터 실행된다.

## 준비

```bash
pip3 install pillow numpy fonttools
apt-get update && apt-get install -y ffmpeg     # ffmpeg + ffprobe
```

폰트는 `scripts/Pretendard-Black.otf` 로 저장소에 함께 들어있다.
다른 한글 볼드(Noto Sans CJK KR Black 등)를 쓰려면 `scripts/common.py` 의
`FONT_PATH` 만 바꾼다.

## 실행

### 0단계 — 인벤토리

```bash
python3 scripts/inventory.py --src <원본폴더> --out 소재
```

내보내는 것:

- `소재/inventory.md` — 사진·영상 표. 방향/해상도/초점/노출/여백은 스크립트가 측정해
  채우고, 「어느 책」·「구도」 칸은 컨택트시트를 눈으로 보고 채운다.
- `소재/contact_photos.jpg`, `소재/contact_videos.jpg` — 눈으로 확인할 컨택트시트.
  영상은 첫 프레임과 좋은 구간 중간 프레임을 뽑아 붙인다.
- `소재/inventory.json`, `소재/inventory.csv` — 기계 판독용.

영상은 프레임을 훑어 **흔들림이 적고 책이 또렷한 구간을 초 단위로** 찾아
`좋은 구간` 칸에 적는다. 그 값을 `cuts.json` 의 `video.start` 에 옮긴다.

### 1단계 — 컷 배정

`cuts.json` 의 `photo` / `video` 를 채운다. 배정 원칙:

| 비트 | 카피 | 배정 |
|---|---|---|
| b1 문제 제기 | 매일 글 쓰는데 / 예약은 왜 안 늘까 | 단권 표지 |
| b2 인식 전환 | 글쓰기가 아니라 / **구조**의 문제였습니다 | 다른 책 표지 또는 내지 |
| b3 해결 제시 | 검색부터 예약까지 / **구조**로 완성합니다 | 두 권이 함께 잡힌 컷 |

카피는 확정본이다. `scripts/common.py` 의 `BEATS` 에 박아두었고 `cuts.json` 으로
덮어쓸 수 없다.

### 2단계 — 정지 소재 (9장)

```bash
python3 scripts/make_stills.py --cuts cuts.json --out 소재/stills
python3 scripts/make_stills.py --cuts cuts.json --beat b2 --ratio 9x16   # 부분 재생성
```

- 1:1 1080×1080 / 4:5 1080×1350 / 9:16 1080×1920, JPEG 품질 92
- 텍스트 좌측 정렬·하단 배치, 좌우 여백 각 8.5%
- 안전 영역: 9:16 상 15%·하 21%, 1:1 상하 7%, 4:5 상하 8%
  (4:5 는 지시문에 명시가 없어 두 값 사이로 잡았다 — `RATIOS` 에서 조정 가능)
- 하단 그라디언트 스크림만 넣고 사진 상단은 건드리지 않는다
- 강조어 `구조` 만 골드 `#FFD166`, 하단에 브랜드명
- 긴 줄은 여백 안에 들어올 때까지 폰트 자동 축소
- 크롭은 책 분포 전체를 담도록 기준점을 잡는다. 중앙 크롭이 아니다.
  손으로 잡으려면 `cuts.json` 의 `anchor`.

### 3단계 — 영상 소재

```bash
python3 scripts/make_video.py --cuts cuts.json --out 소재/video
python3 scripts/make_video.py --cuts cuts.json --ratio 9x16 --keep-work
```

- 9:16 / 1:1 두 벌, 각 12초(비트당 4초)
- `video.start` 구간에서 잘라 쓴다. 파일 앞부분부터 무조건 자르지 않는다.
- 비트별 세그먼트를 각각 만든 뒤 **concat 데믹서**로 조립한다. 자동 편집 필터는 쓰지 않는다.
- 무음(`-an`), H.264 / yuv420p / 30fps / CRF 20 / `+faststart`
- 쓸 만한 구간이 없는 비트는 `photo` 에 Ken Burns 줌 1.12배로 대체한다.

### 4단계 — 검수

```bash
python3 scripts/check.py --stills 소재/stills --video 소재/video
```

규격·한글 글리프 누락·여백/안전 영역 침범·방향·영상 인코딩을 기계가 판정하고,
눈으로 봐야 하는 항목은 목록으로 남긴다. 실패가 있으면 종료 코드 1.

## 반드시 지킨 기술 사항

1. **한글은 전부 코드로 렌더링해서 굽는다.** 영상 자막도 ffmpeg `drawtext` 가 아니라
   PIL 로 만든 투명 PNG 를 `overlay` 로 합성한다 (`common.text_overlay`).
   정지 소재와 영상이 완전히 같은 렌더 경로를 쓴다.
2. **`ImageOps.exif_transpose()` 를 먼저 적용한다** (`common.load_image`).
   안 하면 폰 촬영 JPG 가 90도 눕는다.
3. **자동 편집·자동 하이라이트 도구를 쓰지 않는다.** 세그먼트 수동 조립이 원칙이다.
   0단계의 구간 탐색은 '후보 제시'일 뿐이고, 실제로 쓸 구간은 `cuts.json` 에 사람이 적는다.

## 원본 확보

촬영 원본은 Drive `다미앤마케팅 › 전자책 › 광고소재 › 사진` 에 있다
(`20260704_*` / `20260705_*`, jpg + mp4).

이 세션의 실행 환경에서는 가져올 수 없었다 — 네트워크 정책이 `drive.google.com` 을
막고 있고, 커넥터의 파일 다운로드는 base64 로 대화에 실어오는 방식이라 수백 MB 급
원본에는 쓸 수 없다. 로컬에서 폴더를 내려받아 `--src` 로 넘기면 그대로 돌아간다.

## 출력 구조

```
소재/
  inventory.md
  contact_photos.jpg
  contact_videos.jpg
  stills/  b1_1x1.jpg b1_4x5.jpg b1_9x16.jpg … b3_9x16.jpg
  video/   ad_9x16.mp4  ad_1x1.mp4
```
