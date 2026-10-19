"""
실제 안내판 사진에 OCR(EasyOCR)을 돌려, 위치추정용 '표지판 관측'으로 쓸 수 있는지 평가

  python ocr_eval.py            # data/sign_photos/*.jpg + data/sign_photos_gt.csv

처리
  1) EasyOCR (한국어+영어)로 글자 덩어리 검출·인식
  2) 지도에 있는 표지판 사전(SIGN_ALIASES)과 매칭
       - 숫자만 있는 덩어리: 호실 번호와 '정확히' 일치해야 인정
       - 글자: 유사도(SequenceMatcher) >= 0.7 이면 인정 (OCR 한두 글자 오독 허용)
  3) 사진별 판정
       correct      : 후보가 정답 하나뿐
       ambiguous    : 정답이 후보에 있지만 다른 장소도 후보 (같은 글자가 여러 곳)
       wrong        : 후보가 있는데 정답이 없음 (오독)
       miss         : 아무 후보도 없음
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np

PHOTO_DIR = "data/sign_photos"
GT_CSV = "data/sign_photos_gt.csv"
CACHE = "data/ocr_raw.json"
OUT = "results_ocr"
MIN_CONF = 0.1     # 사전 매칭이 잡음을 걸러주므로 낮게 둠
# 참고: data/ocr_raw.json 은 원본 해상도(2880x2160) 사진으로 만든 캐시. 저장소의 사진은 1600px로 줄였으므로
#       캐시를 지우고 다시 돌리면 결과가 조금 달라질 수 있다.
FUZZY = 0.7

# 지도(signs_1F.csv)의 label  ->  실제 표지판에 적힌 글자들
SIGN_ALIASES = {
    "101": ["101", "하천정보연구실", "River Informatics Lab"], "102": ["102"],
    "103": ["103", "고지능구조기술 연구실", "Laboratory of Intelligent Structure and Technology"], "104": ["104"],
    "105": ["105", "강의실", "Lecture Room"],
    "106": ["106", "세미나실", "Seminar Room", "기계공학과"],
    "107": ["107"], "108": ["108"], "110": ["110"],
    "111": ["111", "에너지환경융합실험실", "Energy & Environment Laboratory"],
    "112": ["112", "강의실", "Lecture Room"],
    "113": ["113"],
    "ME-office": ["기계공학과", "DKU 기계공학과"],
    "EV": ["공과대학", "College of Engineering"],
    "WC(M)": ["남자화장실", "화장실", "Gentlemen"],
    "WC(F)": ["여자화장실", "화장실", "Ladies"],
}
TARGET2LABEL = {"엘리베이터": "EV", "남자화장실": "WC(M)", "여자화장실": "WC(F)", "기계공학과": "ME-office"}


def norm(s):
    return re.sub(r"[^0-9a-z가-힣]", "", s.lower())


def match(text):
    """OCR 글자 하나 -> {label: score}"""
    t = norm(text)
    out = {}
    if not t:
        return out
    if t.isdigit():
        for lab, al in SIGN_ALIASES.items():
            if t in al:
                out[lab] = 1.0
        return out
    if len(t) < 3:
        return out
    for lab, al in SIGN_ALIASES.items():
        for a in al:
            a = norm(a)
            if a.isdigit():
                continue
            r = SequenceMatcher(None, t, a).ratio()
            if a in t:                       # "dku기계공학과" 안에 "기계공학과"
                r = max(r, 0.95)
            if r >= FUZZY:
                out[lab] = max(out.get(lab, 0), r)
    return out


def run_ocr(files):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = [f for f in files if os.path.basename(f) not in cache]
    if todo:
        import easyocr
        reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
        for f in todo:
            res = reader.readtext(f)
            cache[os.path.basename(f)] = [(t, round(float(c), 3), [[int(x), int(y)] for x, y in b]) for b, t, c in res]
            print("OCR", os.path.basename(f), len(res), "tokens", flush=True)
        json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=1)
    return cache


def evaluate():
    gt = {r["image"]: r for r in csv.DictReader(open(GT_CSV, encoding="utf-8"))}
    files = sorted(glob.glob(f"{PHOTO_DIR}/*.jpg"))
    cache = run_ocr(files)
    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        g = gt[name]
        target = TARGET2LABEL.get(g["target"], g["target"])
        strong, weak, hits = defaultdict(float), defaultdict(float), []
        for text, conf, box in cache[os.path.basename(f)]:
            if conf < MIN_CONF:
                continue
            for lab, s in match(text).items():
                # 호실 번호 정확 일치 = 강한 증거, 이름(강의실·세미나실 등) 유사 매칭 = 약한 증거
                (strong if norm(text).isdigit() else weak)[lab] = max(s, (strong if norm(text).isdigit() else weak)[lab])
                hits.append((text, lab, conf, box))
        # 번호를 읽었으면 번호를 믿고, 못 읽었을 때만 이름으로 후보를 만든다
        cands = strong if strong else weak
        number_read = any(norm(t) == target for t, *_ in cache[os.path.basename(f)] if norm(t).isdigit())
        if not cands:
            verdict = "miss"
        elif target in cands and len(cands) == 1:
            verdict = "correct"
        elif target in cands:
            verdict = "ambiguous"
        else:
            verdict = "wrong"
        rows.append(dict(image=name, target=target, dist=float(g["distance_m"]), view=g["view"],
                         angle=int(g["angle_deg"]), verdict=verdict, number_read=number_read,
                         candidates=sorted(cands), evidence="number" if strong else ("name" if weak else "-"), hits=hits,
                         n_tokens=len(cache[os.path.basename(f)])))
    return rows


def summarize(rows):
    os.makedirs(OUT, exist_ok=True)
    lines = ["| image | target | distance | view | evidence | OCR candidates | verdict |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['image']} | {r['target']} | {r['dist']:.0f} m | {r['view']} {r['angle'] or ''} | "
                     f"{r['evidence']} | {', '.join(r['candidates']) or '-'} | **{r['verdict']}** |")

    def rate(sub, key="verdict"):
        n = len(sub)
        c = sum(r[key] in ("correct", "ambiguous") for r in sub) if key == "verdict" else sum(r[key] for r in sub)
        return f"{c}/{n} ({100*c/max(n,1):.0f}%)"

    room = [r for r in rows if r["target"].isdigit()]
    lines += ["", "| group | photos | target found (correct+ambiguous) | uniquely correct | room number read |",
              "|---|---|---|---|---|"]
    groups = [("all", rows), ("room-number signs", room),
              ("frontal", [r for r in rows if r["view"] == "frontal"]),
              ("oblique", [r for r in rows if r["view"] == "oblique"]),
              ("1 m", [r for r in rows if r["dist"] <= 1]),
              ("2-3 m", [r for r in rows if r["dist"] >= 2]),
              ("pictogram / non-room (WC, EV)", [r for r in rows if r["target"] in ("WC(M)", "WC(F)", "EV")])]
    for name, sub in groups:
        rs = [r for r in sub if r["target"].isdigit()]
        lines.append(f"| {name} | {len(sub)} | {rate(sub)} | "
                     f"{sum(r['verdict']=='correct' for r in sub)}/{len(sub)} | "
                     f"{rate(rs, 'number_read') if rs else '-'} |")
    n = len(rows)
    found = sum(r["verdict"] in ("correct", "ambiguous") for r in rows)
    wrong = sum(r["verdict"] == "wrong" for r in rows)
    lines += ["", f"→ 관측 모델 추정치 (n={n}, 표본이 작아 참고용): "
                  f"p_detect ≈ {(found + wrong) / n:.2f}, "
                  f"p_ocr_correct ≈ {found / max(found + wrong, 1):.2f}, "
                  f"ambiguous 비율 ≈ {sum(r['verdict']=='ambiguous' for r in rows)/n:.2f}"]
    txt = "\n".join(lines)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")
    json.dump(rows, open(f"{OUT}/per_image.json", "w"), ensure_ascii=False, indent=1)
    print(txt)


def contact_sheet(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from PIL import Image
    for f in fm.findSystemFonts():
        if "NotoSansCJK-Regular" in f:
            fm.fontManager.addfont(f)
            plt.rcParams["font.family"] = "Noto Sans CJK KR"
    col = {"correct": "#2a9d4f", "ambiguous": "#e9a23b", "wrong": "#d1495b", "miss": "#777777"}
    n = len(rows); c = 4; rr = int(np.ceil(n / c))
    fig, axes = plt.subplots(rr, c, figsize=(4 * c, 3.3 * rr))
    for ax, r in zip(axes.flat, rows):
        im = Image.open(f"{PHOTO_DIR}/{r['image']}.jpg")
        s = 600 / im.width
        ax.imshow(im.resize((600, int(im.height * s))))
        for text, lab, conf, box in r["hits"]:
            b = np.array(box) * s
            ax.add_patch(plt.Polygon(b, fill=False, ec=col["correct" if lab == r["target"] else "wrong"], lw=2))
        ax.set_title(f"{r['image']}  [{r['target']}] {r['dist']:.0f}m {r['view']}\n"
                     f"→ {', '.join(r['candidates']) or '없음'}  ({r['verdict']})",
                     fontsize=9, color=col[r["verdict"]])
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/contact_sheet.jpg", dpi=70)


if __name__ == "__main__":
    rows = evaluate()
    summarize(rows)
    contact_sheet(rows)
