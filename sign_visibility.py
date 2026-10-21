"""
왜 정면을 보고 걷는 영상에서는 호실 번호가 안 읽히는가 — 기하학적 추정

복도 벽에 붙은 번호판은 복도 방향과 '평행'하게 붙어 있다.
정면을 보고 걷는 카메라에서 앞쪽 d m 지점의 번호판은
  - 시야 중심에서 α = atan(lateral / d) 만큼 옆에 보이고
  - 번호판 면을 거의 옆에서(비스듬히) 보게 되어 글자 가로 폭이 sin(α) 배로 찌그러진다.

가정 (근사치):
  카메라 수평 화각 70°, 영상 가로 1280 px (720p)
  번호판 숫자 높이 4 cm, 폭 2.5 cm (사진으로 본 번호판 크기에서 대략 추정)
  카메라 ↔ 벽 거리 1.1 m (복도 폭 약 2.2 m, 중앙 보행)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HFOV, W = np.deg2rad(70), 1280
f = (W / 2) / np.tan(HFOV / 2)
LAT, H_DIGIT, W_DIGIT = 1.1, 0.04, 0.025
WALK = 1.3                     # m/s (영상: 110걸음 × 0.7 m / 약 60 s)

d = np.linspace(0.3, 8, 400)
alpha = np.arctan2(LAT, d)
r = np.hypot(d, LAT)
in_view = alpha < HFOV / 2
h_px = H_DIGIT / r * f
w_px = W_DIGIT * np.sin(alpha) / r * f

# 비교: 정면 1.5 m에서 찍은 사진 (12MP, 가로 4032 px, 같은 화각 가정)
f_photo = (4032 / 2) / np.tan(HFOV / 2)
photo_h = H_DIGIT / 1.5 * f_photo
photo_w = W_DIGIT / 1.5 * f_photo

READABLE = (w_px >= 6) & (h_px >= 10) & in_view
window = d[READABLE]

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(d, np.where(in_view, h_px, np.nan), label="digit height (px)", color="#3a6ea5", lw=2)
ax.plot(d, np.where(in_view, w_px, np.nan), label="digit width (px, foreshortened)", color="#d1495b", lw=2)
ax.axhline(6, color="#d1495b", ls=":", lw=1)
ax.axhline(10, color="#3a6ea5", ls=":", lw=1)
ax.axvspan(0, d[in_view].min(), color="#cccccc", alpha=.5, label="sign outside camera view")
if len(window):
    ax.axvspan(window.min(), window.max(), color="#2a9d4f", alpha=.2, label="maybe readable")
ax.set_xlabel("forward distance to the sign along the corridor (m)")
ax.set_ylabel("pixels in 720p frame")
ax.set_title(f"Forward-facing walking video: door-sign digits are tiny and squashed\n"
             f"(frontal photo at 1.5 m, 12MP: {photo_h:.0f} px tall, {photo_w:.0f} px wide)")
ax.set_ylim(0, 30); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results_video/sign_visibility.png", dpi=130)

if len(window):
    print(f"readable window: d = {window.min():.1f} ~ {window.max():.1f} m "
          f"({(window.max()-window.min())/WALK:.2f} s at {WALK} m/s)")
else:
    print("no readable window under these assumptions")
print(f"frontal photo 1.5 m: {photo_h:.0f} x {photo_w:.0f} px per digit")
