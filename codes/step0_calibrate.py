from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path("E:/Computer Vision")
CALIB_DIR   = PROJECT_DIR / "nerf_data" / "calibration_images"
CALIB_OUT   = PROJECT_DIR / "nerf_data" / "camera_calib.npz"

TAGS_X     = 2        # columns
TAGS_Y     = 3        # rows
TAG_SIZE_M = 0.060    # 60mm
SPACING_H  = 0.090    # 90.00mm  centre-to-centre horizontally
SPACING_V  = 0.07567  # 75.67mm  centre-to-centre vertically

id_to_pos = {}
for row in range(TAGS_Y):
    for col in range(TAGS_X):
        tag_id = row * TAGS_X + col
        id_to_pos[tag_id] = (col, row)

def tag_corners_3d(col, row):
    """4 corners in 3D world (TL,TR,BR,BL) matching ArUco detector order."""
    cx = col * SPACING_H
    cy = row * SPACING_V
    s  = TAG_SIZE_M / 2
    return np.array([
        [cx - s, cy - s, 0],  # TL
        [cx + s, cy - s, 0],  # TR
        [cx + s, cy + s, 0],  # BR
        [cx - s, cy + s, 0],  # BL
    ], dtype=np.float32)

aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

img_paths = sorted(CALIB_DIR.glob("*"))
img_paths = [p for p in img_paths
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
print(f"found {len(img_paths)} calibration images")
print(f"layout: {TAGS_X} cols x {TAGS_Y} rows | tag={TAG_SIZE_M*1000:.0f}mm")
print(f"spacing: H={SPACING_H*1000:.2f}mm  V={SPACING_V*1000:.2f}mm\n")

img_size    = None
all_obj_pts = []
all_img_pts = []

for p in img_paths:
    img = cv2.imread(str(p))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]   # (width, height)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is None or len(ids) < 2:
        print(f"  skip (few tags)): {p.name}")
        continue

    obj_pts_img = []
    img_pts_img = []

    for corner, tag_id in zip(corners, ids.flatten()):
        if tag_id not in id_to_pos:
            continue
        col, row      = id_to_pos[tag_id]
        world_corners = tag_corners_3d(col, row)
        img_corners   = corner.reshape(4, 2)

        for wc, ic in zip(world_corners, img_corners):
            obj_pts_img.append(wc)
            img_pts_img.append(ic)

    if len(obj_pts_img) < 8:
        print(f"  skip(not enough pts): {p.name}")
        continue

    all_obj_pts.append(np.array(obj_pts_img, dtype=np.float32).reshape(-1, 1, 3))
    all_img_pts.append(np.array(img_pts_img, dtype=np.float32).reshape(-1, 1, 2))
    print(f"  ok ({len(ids)} tags, {len(obj_pts_img)} pts): {p.name}")

print(f"\nusing {len(all_obj_pts)} images")

rms, K, dist, _, _ = cv2.calibrateCamera(
    all_obj_pts, all_img_pts, img_size, None, None
)

fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
ratio  = fx / fy

print(f"\nrms:    {rms:.4f} px")
print(f"focal:  fx={fx:.1f}  fy={fy:.1f}  ratio={ratio:.3f}")
print(f"center: cx={cx:.1f}  cy={cy:.1f}")
print(f"dist:   {dist.ravel()[:5]}")

np.savez(CALIB_OUT, camera_matrix=K, dist_coeffs=dist, rms=np.array(rms))
print(f"\nsaved -> {CALIB_OUT}")