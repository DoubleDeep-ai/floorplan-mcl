"""
실제 보행 영상의 OCR 관측으로 MCL 재생 (replay)

  python video_mcl.py --video V_002 --direction LR --steps 110

무엇이 실제이고 무엇이 시뮬레이션인가
  실제      : 표지판 관측 = 영상 프레임(2 fps)에 EasyOCR을 돌려 지도와 매칭한 결과 (video_ocr.py)
  근사 정답 : 위치 = 걸음 수 × 0.7 m를 영상 시간에 일정 속도로 나눔 (말로 기록한 체크포인트가 없어서)
  시뮬레이션: 거리 센서(LiDAR 대신)와 오도메트리 노이즈 — 근사 정답 위치에서 생성

비교하는 세 방법 (같은 거리 센서·오도메트리 노이즈)
  range only       : 표지판 관측 없음
  + simulated OCR  : 사진 22장으로 측정한 확률로 OCR을 흉내 낸 관측 (앞 실험과 동일)
  + real video OCR : 영상에서 실제로 읽힌 표지판
"""
import argparse
import json
import os

import numpy as np

from mcl import Config, FloorMap, ParticleFilter, Robot, raycast

STEP_LEN = 0.7
CORRIDOR_Y = 8.83
X_LEFT, X_RIGHT = 1.5, 77.5      # 복도 중심선의 양 끝 (지도 좌표, m)


def gt_trajectory(times, direction, steps):
    """일정 속도 가정 근사 정답. 걸음 수 거리를 영상 길이에 고르게 나눈다."""
    L = min(steps * STEP_LEN, X_RIGHT - X_LEFT)
    T = times[-1] - times[0]
    s = (times - times[0]) / T * L
    if direction == "LR":
        x, th = X_LEFT + s, 0.0
    else:
        x, th = X_RIGHT - s, np.pi
    return np.c_[x, np.full_like(x, CORRIDOR_Y), np.full_like(x, th)]


class ReplayRobot(Robot):
    """정답 궤적을 따라가는 로봇 (센서 생성용)"""
    def __init__(self, fmap, cfg, rng, poses):
        self.m, self.cfg, self.rng = fmap, cfg, rng
        self.poses, self.k = poses, 0
        self.pose = poses[0].copy()

    def step_to(self, k):
        prev = self.pose
        self.pose = self.poses[k].copy()
        d = np.linalg.norm(self.pose[:2] - prev[:2])
        dth = (self.pose[2] - prev[2] + np.pi) % (2 * np.pi) - np.pi
        c = self.cfg
        return d + self.rng.normal(0, c.sigma_trans * d + 1e-3), dth + self.rng.normal(0, c.sigma_rot)


def run_replay(fmap, cfg, poses, obs_seq, mode, seed):
    rng_world = np.random.default_rng(seed)
    rng_pf = np.random.default_rng(seed + 10_000)
    robot = ReplayRobot(fmap, cfg, rng_world, poses)
    pf = ParticleFilter(fmap, cfg, rng_pf)
    err, mass, frames = [], [], []
    for k in range(1, len(poses)):
        d, dth = robot.step_to(k)
        z = robot.sense_ranges()
        sim_signs = robot.sense_signs()          # 항상 호출 (난수 소비를 같게)
        pf.predict(d, dth)
        pf.update_ranges(z)
        if mode == "sim":
            pf.update_signs(sim_signs)
        elif mode == "real":
            pf.update_signs(obs_seq[k])
        est = pf.estimate()
        err.append(np.linalg.norm(est - robot.pose[:2]))
        mass.append(pf.mass_near(robot.pose[:2]))
        frames.append((pf.p[:, :2].copy(), pf.w.copy(), est))
        pf.resample()
    return np.array(err), np.array(mass), frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--direction", choices=["LR", "RL"], required=True)
    ap.add_argument("--steps", type=float, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--particles", type=int, default=6000)
    ap.add_argument("--out", default="results_video")
    a = ap.parse_args()

    meta = json.load(open("data/plan_clean.json"))
    fmap = FloorMap.from_image("data/plan_clean.png", width_m=meta["width_m"], signs_csv="data/signs_1F_ocr.csv")
    rows = json.load(open(f"{a.out}/{a.video}_ocr.json"))
    times = np.array([r["t"] for r in rows])
    poses = gt_trajectory(times, a.direction, a.steps)
    obs = [r["labels"] for r in rows]
    cfg_kw = dict(n_particles=a.particles, sign_view_dist=3.0, p_detect=0.9, p_ocr_correct=0.95)

    res = {}
    for mode in ("none", "sim", "real"):
        E, M = [], []
        for s in range(a.seeds):
            e, m, _ = run_replay(fmap, Config(use_signs=(mode != "none"), **cfg_kw), poses, obs, mode, s)
            E.append(e); M.append(m)
        res[mode] = (np.array(E), np.array(M))
        loc = [np.nonzero(m > 0.5)[0] for m in M]
        first = [l[0] if len(l) else np.nan for l in loc]
        print(f"{a.video} {mode:5s}: final err median {np.median([e[-10:].mean() for e in E]):.2f} m | "
              f"localized(>0.5 mass) at step median {np.nanmedian(first):.0f} "
              f"| success {np.mean([m[-10:].mean() > 0.5 for m in M])*100:.0f}%", flush=True)
    np.savez(f"{a.out}/{a.video}_replay.npz", times=times, poses=poses,
             **{f"{k}_err": v[0] for k, v in res.items()}, **{f"{k}_mass": v[1] for k, v in res.items()})
