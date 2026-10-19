"""
피난안내도 사진 → MCL용 깨끗한 도면(검은 벽 / 흰 자유공간) 변환

  python floorplan_from_photo.py photo.jpg --crop 619 876 2307 1289 --steps 113

처리 순서
  1) 도면 영역 crop
  2) 벽 픽셀 추출: 어둡고(V<105) 무채색이며 초록(복도 칠)·파랑(현위치 표시)이 아닌 픽셀
  3) 작은 덩어리 제거: 방 번호 상자, 글자, 아이콘 (가로·세로 100 px 미만) + 얇은 선 제거(opening)
  4) 기울기 보정 (윗벽이 수평이 되도록 회전)
  5) 건물 바깥을 벽으로 채움 (파티클이 건물 밖에 뿌려지지 않게)
  6) 축척: 복도 끝~끝 걸음 수 × 보폭 → m/pixel
"""
import argparse
import json

import cv2
import numpy as np


def extract_walls(crop, min_size=100):
    c = crop.astype(int)
    B, G, R = c[..., 0], c[..., 1], c[..., 2]
    V, chroma = c.max(-1), c.max(-1) - c.min(-1)
    green = (G - np.maximum(R, B)) > 8          # 복도 초록 칠 (그늘져서 어두워도 제외)
    blue = (B - np.maximum(R, G)) > 15          # 현위치 표시선
    dark = (V < 105) & (chroma < 45) & ~green & ~blue
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    keep = [i for i in range(1, n) if max(st[i, 2], st[i, 3]) >= min_size]
    walls = np.isin(lab, keep).astype(np.uint8)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # 얇은 선(소화전·소화기 아이콘 테두리 등, 두께 <4 px) 제거. 실제 벽은 5 px 이상
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    return walls.astype(bool)


def deskew(walls):
    cols = np.arange(int(walls.shape[1] * 0.05), int(walls.shape[1] * 0.95), 10)
    top = np.array([np.argmax(walls[:, c]) for c in cols])
    slope = np.polyfit(cols, top, 1)[0]
    ang = np.degrees(np.arctan(slope))
    h, w = walls.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    out = cv2.warpAffine(walls.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST)
    return out.astype(bool), ang, M


def fill_exterior(walls, min_room_px):
    ys, xs = np.nonzero(walls)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    occ = walls.copy()
    occ[:y0], occ[y1 + 1:], occ[:, :x0], occ[:, x1 + 1:] = True, True, True, True
    # 건물 외곽 박스를 닫아 출입구로 바깥이 새어 들어오지 않게 함
    occ[y0, x0:x1 + 1] = occ[y1, x0:x1 + 1] = True
    occ[y0:y1 + 1, x0] = occ[y0:y1 + 1, x1] = True
    # 작은 빈 공간(문 상자 속, 외곽 틈)은 벽으로
    n, lab, st, _ = cv2.connectedComponentsWithStats((~occ).astype(np.uint8), 4)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_room_px:
            occ[lab == i] = True
    return occ[y0:y1 + 1, x0:x1 + 1], (x0, y0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--crop", type=int, nargs=4, required=True, help="x0 y0 x1 y1 (원본 사진 픽셀)")
    ap.add_argument("--steps", type=float, required=True, help="복도 끝~끝 걸음 수")
    ap.add_argument("--step-len", type=float, default=0.7, help="보폭 (m)")
    ap.add_argument("--out", default="data/plan_clean.png")
    a = ap.parse_args()

    img = cv2.imread(a.photo)
    x0, y0, x1, y1 = a.crop
    crop = img[y0:y1, x0:x1]
    walls = extract_walls(crop)
    walls, ang, M = deskew(walls)

    # 축척: 복도 = 건물 왼쪽 끝 ~ 오른쪽 끝이라고 가정
    ys, xs = np.nonzero(walls)
    corridor_px = xs.max() - xs.min()
    m_per_px = a.steps * a.step_len / corridor_px
    occ, offset = fill_exterior(walls, min_room_px=int(0.5 / m_per_px ** 2))

    cv2.imwrite(a.out, np.where(occ, 0, 255).astype(np.uint8))
    meta = {"photo": a.photo, "crop": a.crop, "rotation_deg": float(ang), "offset_px": [int(v) for v in offset],
            "m_per_px": m_per_px, "width_m": occ.shape[1] * m_per_px, "height_m": occ.shape[0] * m_per_px,
            "steps": a.steps, "step_len": a.step_len}
    json.dump(meta, open(a.out.replace(".png", ".json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))
