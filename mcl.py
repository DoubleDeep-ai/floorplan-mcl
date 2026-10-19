"""
도면(Floor plan) 위 Monte Carlo Localization (Particle filter)
 + 표지판 텍스트(OCR) 관측 결합

구성
  FloorMap        : 도면 → 점유 격자(occupancy grid). 합성 지도 생성 / 도면 이미지 로딩
  raycast()       : 격자 위 레이캐스팅 (거리 센서 시뮬레이션 + 파티클의 예상 거리)
  ParticleFilter  : 예측(모션) → 가중치 갱신(거리 센서, 표지판) → 리샘플링
  Robot           : 정답 궤적 + 노이즈 섞인 오도메트리/센서/OCR 관측 생성

모든 좌표 단위는 m, 각도는 rad.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# =====================================================================
# 지도
# =====================================================================
@dataclass
class FloorMap:
    occ: np.ndarray                 # (H, W) bool, True = 벽/장애물
    res: float                      # m / pixel
    signs: dict = field(default_factory=dict)   # label -> [(x, y), ...] [m]
                                                # 같은 글자가 여러 곳에 있을 수 있음 (예: 양쪽 끝 '화장실')

    @property
    def size(self):
        H, W = self.occ.shape
        return W * self.res, H * self.res     # (width_m, height_m)

    def is_free(self, xy: np.ndarray) -> np.ndarray:
        """xy: (..., 2) [m] → 자유 공간 여부 (지도 밖은 False)"""
        ix = np.floor(xy[..., 0] / self.res).astype(int)
        iy = np.floor(xy[..., 1] / self.res).astype(int)
        H, W = self.occ.shape
        inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        out = np.zeros(ix.shape, dtype=bool)
        out[inside] = ~self.occ[iy[inside], ix[inside]]
        return out

    def sample_free(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """자유 공간에서 균일하게 n개 위치 샘플 → (n, 2)"""
        iy, ix = np.nonzero(~self.occ)
        k = rng.integers(0, len(ix), n)
        jitter = rng.random((n, 2))
        return np.c_[ix[k] + jitter[:, 0], iy[k] + jitter[:, 1]] * self.res

    # ---------------- 합성 지도 ----------------
    @staticmethod
    def synthetic_symmetric(res: float = 0.1) -> "FloorMap":
        """
        좌우·상하 대칭인 가상 건물 한 층 (40 m x 20 m).
        긴 복도 + 양 끝 세로 복도 + 위아래 강의실.
        → 거리 센서만으로는 대칭 위치를 구별할 수 없는 '최악의 경우'.
        표지판(호실 번호)만 위치마다 다르다.
        """
        W, H = int(40 / res), int(20 / res)
        occ = np.ones((H, W), dtype=bool)

        def free(x0, x1, y0, y1):
            occ[round(y0 / res):round(y1 / res), round(x0 / res):round(x1 / res)] = False

        free(2, 38, 9, 11)          # 가로 메인 복도 (폭 2 m)
        free(2, 4, 2, 18)           # 왼쪽 세로 복도
        free(36, 38, 2, 18)         # 오른쪽 세로 복도

        signs = {}
        room_x = [6, 11, 16, 20.5, 25.5, 30.5]  # 강의실 6개씩 (중심 x=20 기준 좌우 대칭)
        for i, x0 in enumerate(room_x):
            w = 3.5
            free(x0, x0 + w, 12, 17)            # 위쪽 강의실
            free(x0, x0 + w, 3, 8)              # 아래쪽 강의실
            dx = x0 + 0.5 if i < 3 else x0 + w - 1.5      # 문 위치도 좌우 대칭
            free(dx, dx + 1.0, 11, 12)          # 위쪽 문
            free(dx, dx + 1.0, 8, 9)            # 아래쪽 문
            signs[f"3{i+1:02d}"] = [(dx + 0.5, 11.0)]     # 301~306 (위)
            signs[f"3{i+51:02d}"] = [(dx + 0.5, 9.0)]     # 351~356 (아래)
        signs["EV"] = [(3.0, 17.5)]       # 엘리베이터 (왼쪽 위만 있음)
        signs["WC"] = [(37.0, 2.5)]       # 화장실 (오른쪽 아래만 있음)
        return FloorMap(occ=occ, res=res, signs=signs)

    # ---------------- 도면 이미지 로딩 ----------------
    @staticmethod
    def from_image(path: str, width_m: float, signs_csv: str | None = None,
                   threshold: int = 128, res: float = 0.1) -> "FloorMap":
        """
        도면 사진(피난안내도 등)을 점유 격자로 변환.
          path      : 흰 바탕에 검은 벽으로 정리한 도면 이미지
          width_m   : 이미지 가로가 실제로 몇 m인지 (축척)
          signs_csv : 'label,x_px,y_px' 형식 (도면 이미지 픽셀 좌표)
        """
        from PIL import Image
        img = Image.open(path).convert("L")
        scale = width_m / img.width                  # m / 원본 pixel
        new_w = int(round(width_m / res))
        new_h = int(round(img.height * scale / res))
        g = np.asarray(img.resize((new_w, new_h), Image.NEAREST))
        occ = g < threshold                          # 어두운 픽셀 = 벽
        occ = occ[::-1]                              # 이미지 y축(아래로) → 지도 y축(위로)
        signs = {}
        if signs_csv:
            import csv
            with open(signs_csv, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    x = float(row["x"]) * scale
                    y = (img.height - float(row["y"])) * scale
                    signs.setdefault(row["label"].strip(), []).append((x, y))
        return FloorMap(occ=occ, res=res, signs=signs)


# =====================================================================
# 레이캐스팅
# =====================================================================
def raycast(fmap: FloorMap, pose: np.ndarray, beam_angles: np.ndarray,
            max_range: float) -> np.ndarray:
    """
    pose: (N, 3) [x, y, theta], beam_angles: (B,) 로봇 기준 각도
    → (N, B) 각 빔이 벽에 닿는 거리 (못 닿으면 max_range)
    """
    step = fmap.res          # 벽 두께가 1 pixel이면 0.5*res 권장 (느려짐)
    s = np.arange(step, max_range + step, step)                       # (S,)
    ang = pose[:, 2:3] + beam_angles[None, :]                          # (N, B)
    px = pose[:, 0, None, None] + np.cos(ang)[..., None] * s           # (N, B, S)
    py = pose[:, 1, None, None] + np.sin(ang)[..., None] * s
    hit = ~fmap.is_free(np.stack([px, py], -1))                        # (N, B, S)
    first = np.where(hit.any(-1), hit.argmax(-1), len(s) - 1)
    return s[first]


def line_of_sight(fmap: FloorMap, a: np.ndarray, b: np.ndarray) -> bool:
    """두 점 사이에 벽이 없는지 (표지판이 보이는지)"""
    n = max(int(np.linalg.norm(b - a) / (fmap.res * 0.5)), 2)
    t = np.linspace(0, 1, n)[1:-1, None]
    pts = a + (b - a) * t
    # 표지판은 벽에 붙어 있으므로 끝 0.3 m는 검사에서 제외
    keep = np.linalg.norm(pts - b, axis=1) > 0.3
    return bool(fmap.is_free(pts[keep]).all())


# =====================================================================
# 설정
# =====================================================================
@dataclass
class Config:
    n_particles: int = 2000
    n_beams: int = 12
    max_range: float = 8.0
    # 모션 노이즈 (오도메트리 오차)
    sigma_trans: float = 0.10       # 이동량 비례 (m/m)
    sigma_rot: float = 0.05         # rad
    # 거리 센서 모델
    sigma_range: float = 0.4        # m
    p_rand_range: float = 0.1       # 엉뚱한 측정 (사람, 유리문 등) 비율
    range_temp: float = 0.2         # 빔들이 서로 독립이 아니므로 likelihood를 완화(tempering).
                                    # 1.0이면 과신 → 초반에 파티클이 몇 개로 붕괴(particle depletion)
    # 표지판(OCR) 관측 모델
    use_signs: bool = True
    sign_view_dist: float = 4.0     # 로봇이 이 거리 안의 표지판만 읽을 수 있음
    p_detect: float = 0.7           # 보이는 표지판을 실제로 읽을 확률
    p_ocr_correct: float = 0.85     # 읽었을 때 맞게 읽을 확률 (틀리면 다른 호실로 오독)
    sign_sigma: float = 1.0         # 표지판 likelihood 경계의 완만함 (m)
    sign_eps: float = 0.05          # OCR 오독 대비 최소 likelihood (과신 방지)
    # 강건성
    p_inject: float = 0.01          # 매 스텝 무작위 파티클 주입 비율 (복구용)

    @property
    def beam_angles(self):
        return np.linspace(-np.pi, np.pi, self.n_beams, endpoint=False)


# =====================================================================
# 정답 로봇 (시뮬레이션)
# =====================================================================
class Robot:
    def __init__(self, fmap: FloorMap, waypoints, cfg: Config, rng, speed=0.5):
        self.m, self.cfg, self.rng = fmap, cfg, rng
        self.wps = [np.array(w, float) for w in waypoints]
        p0 = self.wps[0]
        h0 = np.arctan2(*(self.wps[1] - p0)[::-1])
        self.pose = np.array([p0[0], p0[1], h0])
        self.k, self.speed = 1, speed

    def done(self):
        return self.k >= len(self.wps)

    def step(self):
        """웨이포인트를 따라 한 스텝 이동 → (오도메트리 관측 (d, dtheta))"""
        target = self.wps[self.k]
        vec = target - self.pose[:2]
        dist = np.linalg.norm(vec)
        new_h = np.arctan2(vec[1], vec[0])
        d = min(self.speed, dist)
        dth = (new_h - self.pose[2] + np.pi) % (2 * np.pi) - np.pi
        self.pose = np.array([self.pose[0] + d * np.cos(new_h),
                              self.pose[1] + d * np.sin(new_h), new_h])
        if dist <= self.speed + 1e-9:
            self.k += 1
        c = self.cfg   # 오도메트리에는 노이즈가 섞여 있음
        d_odo = d + self.rng.normal(0, c.sigma_trans * d + 1e-3)
        dth_odo = dth + self.rng.normal(0, c.sigma_rot)
        return d_odo, dth_odo, dth

    def sense_ranges(self):
        c = self.cfg
        z = raycast(self.m, self.pose[None], c.beam_angles, c.max_range)[0]
        z = z + self.rng.normal(0, c.sigma_range * 0.5, z.shape)
        rnd = self.rng.random(z.shape) < c.p_rand_range * 0.5
        z[rnd] = self.rng.uniform(0, c.max_range, rnd.sum())
        return np.clip(z, 0, c.max_range)

    def sense_signs(self):
        """보이는 표지판을 OCR로 읽은 결과 (오독 포함) → label 리스트"""
        c, out = self.cfg, []
        labels = list(self.m.signs)
        for lab, p in ((l, np.array(q)) for l, ps in self.m.signs.items() for q in ps):
            if np.linalg.norm(p - self.pose[:2]) > c.sign_view_dist:
                continue
            if not line_of_sight(self.m, self.pose[:2], p):
                continue
            if self.rng.random() > c.p_detect:
                continue
            if self.rng.random() < c.p_ocr_correct:
                out.append(lab)
            else:                                   # 오독: 다른 호실 번호로 읽음
                out.append(self.rng.choice([l for l in labels if l != lab]))
        return out


# =====================================================================
# Particle filter
# =====================================================================
class ParticleFilter:
    def __init__(self, fmap: FloorMap, cfg: Config, rng):
        self.m, self.cfg, self.rng = fmap, cfg, rng
        N = cfg.n_particles
        xy = fmap.sample_free(N, rng)                 # 전역 위치추정: 어디인지 모름
        th = rng.uniform(-np.pi, np.pi, N)
        self.p = np.c_[xy, th]                         # (N, 3)
        self.w = np.full(N, 1.0 / N)
        self.sign_pos = {l: np.array(ps, float) for l, ps in fmap.signs.items()}   # label -> (K, 2)

    # 1) 예측: 오도메트리만큼 이동 + 노이즈
    def predict(self, d, dth):
        c, N = self.cfg, len(self.p)
        th = self.p[:, 2] + dth + self.rng.normal(0, c.sigma_rot, N)
        dd = d + self.rng.normal(0, c.sigma_trans * abs(d) + 1e-3, N)
        self.p[:, 0] += dd * np.cos(th)
        self.p[:, 1] += dd * np.sin(th)
        self.p[:, 2] = (th + np.pi) % (2 * np.pi) - np.pi
        self.w[~self.m.is_free(self.p[:, :2])] *= 1e-6     # 벽 안으로 들어간 파티클

    # 2-a) 거리 센서 갱신
    def update_ranges(self, z):
        c = self.cfg
        zhat = raycast(self.m, self.p, c.beam_angles, c.max_range)   # (N, B)
        g = np.exp(-0.5 * ((z - zhat) / c.sigma_range) ** 2) / (c.sigma_range * np.sqrt(2 * np.pi))
        lik = (1 - c.p_rand_range) * g + c.p_rand_range / c.max_range
        self._apply(c.range_temp * np.log(lik).sum(1))

    # 2-b) 표지판(OCR) 갱신
    def update_signs(self, labels):
        """
        'label을 읽었다' = '그 표지판이 view_dist 안에 있다'.
        파티클 위치에서 그 표지판까지 거리가 view_dist 밖이면 likelihood가 부드럽게 감소.
        sign_eps: OCR 오독 가능성 → 한 번의 관측으로 파티클이 전멸하지 않게 함.
        """
        c = self.cfg
        if not labels:
            return
        ll = np.zeros(len(self.p))
        for lab in labels:
            if lab not in self.sign_pos:
                continue
            S = self.sign_pos[lab]                                       # 같은 글자의 모든 위치
            dist = np.linalg.norm(self.p[:, None, :2] - S[None], axis=2).min(1)   # 가장 가까운 것 기준
            over = np.maximum(dist - c.sign_view_dist, 0)
            ll += np.log(c.sign_eps + (1 - c.sign_eps) * np.exp(-0.5 * (over / c.sign_sigma) ** 2))
        self._apply(ll)

    def _apply(self, loglik):
        lw = np.log(self.w + 1e-300) + loglik
        lw -= lw.max()
        w = np.exp(lw)
        self.w = w / w.sum()

    # 3) 리샘플링 (low-variance) + 무작위 파티클 주입
    def resample(self):
        N, c = len(self.p), self.cfg
        neff = 1.0 / np.sum(self.w ** 2)
        if neff < N / 2:
            pos = (self.rng.random() + np.arange(N)) / N
            idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), N - 1)
            self.p = self.p[idx]
            self.w = np.full(N, 1.0 / N)
        k = int(c.p_inject * N)
        if k:
            j = self.rng.choice(N, k, replace=False)
            self.p[j, :2] = self.m.sample_free(k, self.rng)
            self.p[j, 2] = self.rng.uniform(-np.pi, np.pi, k)
            self.w[j] = self.w.mean()
            self.w /= self.w.sum()
        return neff

    def estimate(self):
        """가중 평균 위치 (다봉 분포면 엉뚱한 중간 지점이 나올 수 있음)"""
        return (self.w[:, None] * self.p[:, :2]).sum(0)

    def mass_near(self, xy, r=1.0):
        """정답 위치 r m 안에 있는 확률 질량 → 위치추정 '확신' 지표"""
        return self.w[np.linalg.norm(self.p[:, :2] - xy, axis=1) < r].sum()


# =====================================================================
# 한 번의 실행
# =====================================================================
DEFAULT_ROUTE = [(12, 10), (34, 10), (37, 10), (37, 16), (37, 4), (37, 10), (28, 10)]


def run(fmap: FloorMap, cfg: Config, seed: int, route=DEFAULT_ROUTE, record=False):
    """
    같은 seed면 정답 궤적·센서 노이즈가 같음 → use_signs만 바꿔 공정 비교 가능.
    """
    rng_world = np.random.default_rng(seed)          # 로봇/센서
    rng_pf = np.random.default_rng(seed + 10_000)    # 필터
    robot = Robot(fmap, route, cfg, rng_world)
    pf = ParticleFilter(fmap, cfg, rng_pf)
    hist = {"err": [], "mass": [], "true": [], "est": [], "signs": [], "frames": []}
    while not robot.done():
        d, dth, _ = robot.step()
        z = robot.sense_ranges()
        seen = robot.sense_signs()                   # 항상 같은 난수 소비 (공정 비교)
        pf.predict(d, dth)
        pf.update_ranges(z)
        if cfg.use_signs:
            pf.update_signs(seen)
        est = pf.estimate()
        hist["err"].append(np.linalg.norm(est - robot.pose[:2]))
        hist["mass"].append(pf.mass_near(robot.pose[:2]))
        hist["true"].append(robot.pose.copy())
        hist["est"].append(est)
        hist["signs"].append(seen)
        if record:
            hist["frames"].append((pf.p[:, :2].copy(), pf.w.copy()))
        pf.resample()
    return {k: (np.array(v) if k in ("err", "mass", "true", "est") else v) for k, v in hist.items()}
