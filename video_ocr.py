"""
복도 보행 영상 → 프레임별 OCR → 지도 표지판 매칭 (실제 관측 시퀀스 만들기)

  python video_ocr.py data/video/V_002.mp4 --fps 2

출력: results_video/<영상이름>_ocr.json
  [{"t": 초, "frame": 번호, "labels": ["105", ...], "hits": [[글자, label, conf, box], ...]}, ...]

매칭 규칙은 ocr_eval.py 와 동일 (번호 정확 일치 우선, 이름은 유사도 매칭).
"""
import argparse
import json
import os

import cv2

from ocr_eval import MIN_CONF, match, norm


def extract_frames(path, fps):
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(round(src_fps / fps)), 1)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i / src_fps, i, frame
        i += 1
    cap.release()


def frame_labels(results):
    strong, weak, hits = {}, {}, []
    for text, conf, box in results:
        if conf < MIN_CONF:
            continue
        for lab, s in match(text).items():
            d = strong if norm(text).isdigit() else weak
            d[lab] = max(s, d.get(lab, 0))
            hits.append((text, lab, conf, box))
    return sorted(strong if strong else weak), hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", default="results_video")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.video))[0]
    cache_path = f"{a.out}/{name}_ocr_raw.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    import easyocr
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    rows = []
    for t, i, frame in extract_frames(a.video, a.fps):
        key = str(i)
        if key not in cache:
            res = reader.readtext(frame[:, :, ::-1])     # BGR -> RGB
            cache[key] = [(txt, round(float(c), 3), [[int(x), int(y)] for x, y in b]) for b, txt, c in res]
            json.dump(cache, open(cache_path, "w"), ensure_ascii=False)
        labels, hits = frame_labels(cache[key])
        rows.append({"t": round(t, 2), "frame": i, "labels": labels, "hits": hits,
                     "n_tokens": len(cache[key])})
        if labels:
            print(f"{name} t={t:5.1f}s  {labels}", flush=True)
    json.dump(rows, open(f"{a.out}/{name}_ocr.json", "w"), ensure_ascii=False, indent=1)
    print(f"{name}: {len(rows)} frames, {sum(bool(r['labels']) for r in rows)} with sign matches")
