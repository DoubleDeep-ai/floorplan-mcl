"""
실험: 거리 센서만 쓴 MCL vs 거리 센서 + 표지판(OCR) 관측을 쓴 MCL

  python experiment.py --seeds 20            # 비교 실험 + 그래프
  python experiment.py --gif --gif-seed 2    # 나란히 비교 애니메이션

  # 실제 도면 (피난안내도에서 변환한 지도)
  python experiment.py --plan data/plan_clean.png --signs data/signs_1F.csv \
      --route "25,8.83;73,8.83;40,8.83" --out results_1F --particles 6000 --seeds 10

같은 seed에서는 정답 궤적·센서 노이즈·OCR 오독이 완전히 같고, use_signs만 다르다.
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from mcl import FloorMap, Config, run, DEFAULT_ROUTE

OUT = "results"
ROUTE = DEFAULT_ROUTE
SIGN_KW = {}          # --sign-params 로 표지판 관측 모델 덮어쓰기 (OCR 실측값)
SUCCESS_MASS = 0.5     # 정답 1 m 안에 확률 질량 50% 이상이면 '위치를 찾았다'


def draw_map(ax, fmap, signs=True):
    w, h = fmap.size
    ax.imshow(fmap.occ, origin="lower", extent=[0, w, 0, h], cmap="Greys", vmin=0, vmax=1.4)
    if signs:
        mid = fmap.size[1] / 2
        for lab, ps in fmap.signs.items():
            for x, y in ps:
                ax.plot(x, y, "^", color="#d1495b", ms=5)
                ax.text(x, y + (0.4 if y >= mid else -0.9), lab, fontsize=6, color="#d1495b", ha="center")
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def plot_map(fmap):
    w, h = fmap.size
    fig, ax = plt.subplots(figsize=(12, 12 * h / w + 1))
    draw_map(ax, fmap)
    r = np.array(ROUTE)
    ax.plot(r[:, 0], r[:, 1], "--", color="#3a6ea5", lw=1.5, label="robot route")
    ax.plot(*r[0], "o", color="#3a6ea5", ms=8, label="start")
    ax.set_title("Floor map. Red = signs (room numbers), blue = robot route")
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{OUT}/map.png", dpi=130); plt.close(fig)


def compare(fmap, seeds, n_particles):
    res = {False: [], True: []}
    for us in (False, True):
        for s in seeds:
            t = time.time()
            h = run(fmap, Config(use_signs=us, n_particles=n_particles, **SIGN_KW), seed=s, route=ROUTE)
            res[us].append(h)
            ok = h["mass"][-20:].mean() > SUCCESS_MASS
            print(f"signs={us!s:5} seed={s:2d} final_err={h['err'][-20:].mean():5.2f} m "
                  f"success={ok}  ({time.time()-t:.0f}s)", flush=True)
    return res


def summarize(res):
    rows = []
    for us, name in ((False, "range only"), (True, "range + OCR signs")):
        H = res[us]
        succ = np.array([h["mass"][-20:].mean() > SUCCESS_MASS for h in H])
        conv = []
        for h in H:
            hit = np.nonzero(h["mass"] > SUCCESS_MASS)[0]
            conv.append(hit[0] if len(hit) else np.nan)
        conv = np.array(conv, float)
        final = np.array([h["err"][-20:].mean() for h in H])
        rows.append((name, 100 * succ.mean(), np.nanmedian(conv[succ]) if succ.any() else np.nan,
                     np.median(final), np.median(final[succ]) if succ.any() else np.nan))
    lines = ["| method | success rate | median steps to localize* | median final error (all) | median final error (successful) |",
             "|---|---|---|---|---|"]
    for n, s, c, f, fs in rows:
        lines.append(f"| {n} | {s:.0f}% | {c:.0f} ({c*0.5:.1f} m) | {f:.2f} m | {fs:.2f} m |")
    lines.append("")
    lines.append(f"\\* localized = probability mass within 1 m of the true position > {SUCCESS_MASS}. "
                 f"success = this holds over the last 20 steps. {len(res[False])} seeds each.")
    txt = "\n".join(lines)
    open(f"{OUT}/summary.md", "w").write(txt + "\n")
    print(txt)


def plot_curves(res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for us, name, col in ((False, "range only", "#999999"), (True, "range + OCR signs", "#3a6ea5")):
        L = min(len(h["err"]) for h in res[us])
        E = np.array([h["err"][:L] for h in res[us]])
        M = np.array([h["mass"][:L] for h in res[us]])
        x = np.arange(L) * 0.5          # 이동 거리 (m)
        axes[0].plot(x, np.median(E, 0), color=col, lw=2, label=name)
        axes[0].fill_between(x, np.percentile(E, 25, 0), np.percentile(E, 75, 0), color=col, alpha=.2)
        axes[1].plot(x, M.mean(0), color=col, lw=2, label=name)
    axes[0].set_ylabel("position error (m)"); axes[0].set_title("Error of estimate (median, IQR)")
    axes[1].set_ylabel("prob. mass within 1 m of truth"); axes[1].set_title("Confidence in the TRUE position (mean)")
    for a in axes:
        a.set_xlabel("distance travelled (m)"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/curves.png", dpi=130); plt.close(fig)


def make_gif(fmap, seed, n_particles, every=2):
    H = {us: run(fmap, Config(use_signs=us, n_particles=n_particles, **SIGN_KW), seed=seed, route=ROUTE, record=True)
         for us in (False, True)}
    w, h = fmap.size
    fig, axes = plt.subplots(2, 1, figsize=(12, 2 * 12 * h / w + 1.2)) if w / h > 3 else plt.subplots(1, 2, figsize=(13, 3.9))
    rng = np.random.default_rng(0)
    for a, us in zip(axes, (False, True)):
        draw_map(a, fmap)
    scat = [a.scatter([], [], s=2, c="#f2a65a", alpha=.6) for a in axes]
    true = [a.plot([], [], "o", color="#2a9d4f", ms=7, label="true")[0] for a in axes]
    est = [a.plot([], [], "x", color="k", ms=8, mew=2, label="estimate")[0] for a in axes]
    txt = [a.text(0.5, -0.06, "", transform=a.transAxes, ha="center", fontsize=8) for a in axes]
    axes[0].legend(loc="upper center", fontsize=7, ncol=2)
    T = len(H[False]["err"])

    def upd(k):
        t = k * every
        for i, us in enumerate((False, True)):
            h = H[us]
            P, W = h["frames"][t]
            j = rng.choice(len(P), 1500, p=W)          # 가중치대로 뽑아서 표시
            scat[i].set_offsets(P[j])
            true[i].set_data([h["true"][t][0]], [h["true"][t][1]])
            est[i].set_data([h["est"][t][0]], [h["est"][t][1]])
            seen = ", ".join(map(str, h["signs"][t])) if us else "-"
            axes[i].set_title(f"{'range + OCR signs' if us else 'range only'}   "
                              f"step {t}  err {h['err'][t]:.1f} m", fontsize=9)
            txt[i].set_text(f"OCR read: {seen}" if us else "OCR not used")
        return scat + true + est + txt

    anim = FuncAnimation(fig, upd, frames=T // every, blit=False)
    anim.save(f"{OUT}/mcl_compare.gif", writer=PillowWriter(fps=6), dpi=80)
    plt.close(fig)

    # 핵심 장면 정지 그림 (README용)
    idx = [0, 3, 12, T - 1]
    fig, axes = plt.subplots(2, len(idx), figsize=(16, 2.8 if w / h > 3 else 4.6))
    for c, t in enumerate(idx):
        for r, us in enumerate((False, True)):
            a = axes[r, c]; draw_map(a, fmap, signs=False)
            P, W = H[us]["frames"][t]
            j = rng.choice(len(P), 1500, p=W)
            a.scatter(P[j, 0], P[j, 1], s=1.5, c="#f2a65a", alpha=.6)
            a.plot(*H[us]["true"][t][:2], "o", color="#2a9d4f", ms=6)
            a.set_title(f"{'+OCR' if us else 'range only'}  step {t}", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/snapshots.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--particles", type=int, default=3000)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif-seed", type=int, default=2)
    ap.add_argument("--plan", help="도면 이미지 (floorplan_from_photo.py 결과). 없으면 합성 대칭 지도")
    ap.add_argument("--signs", help="label,x,y CSV (도면 이미지 픽셀 좌표)")
    ap.add_argument("--route", help='웨이포인트 "x,y;x,y;..." (m)')
    ap.add_argument("--out", default="results")
    ap.add_argument("--sign-params", help='예: "sign_view_dist=3,p_detect=0.9,p_ocr_correct=0.95"')
    a = ap.parse_args()
    if a.sign_params:
        SIGN_KW = {k: float(v) for k, v in (kv.split("=") for kv in a.sign_params.split(","))}
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    if a.plan:
        meta = json.load(open(a.plan.replace(".png", ".json")))
        fmap = FloorMap.from_image(a.plan, width_m=meta["width_m"], signs_csv=a.signs)
    else:
        fmap = FloorMap.synthetic_symmetric()
    if a.route:
        ROUTE = [tuple(map(float, p.split(","))) for p in a.route.split(";")]
    plot_map(fmap)
    if a.gif:
        make_gif(fmap, a.gif_seed, a.particles)
    else:
        res = compare(fmap, range(a.seeds), a.particles)
        summarize(res)
        plot_curves(res)
