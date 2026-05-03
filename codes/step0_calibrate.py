from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path("E:/Computer Vision")

# config - point CALIB_DIR to your 30-50 aruco grid photos
CALIB_DIR  = PROJECT_DIR / "nerf_data" / "calibration_images"
CALIB_OUT  = PROJECT_DIR / "nerf_data" / "camera_calib.npz"
TAG_SIZE_M = 0.025   # physical size of one aruco tag in metres (measure your printout)

aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

all_obj_pts = []
all_img_pts = []
img_size    = None

img_paths = sorted(CALIB_DIR.glob("*"))
img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
print(f"found {len(img_paths)} calibration images")

for p in img_paths:
    img  = cv2.imread(str(p))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img_size is None:
        img_size = gray.shape[::-1]

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is None or len(ids) == 0:
        print(f"  skip (no tags): {p.name}")
        continue

    # tag corners in order: TL, TR, BR, BL
    for corner, tag_id in zip(corners, ids.flatten()):
        s = TAG_SIZE_M
        obj = np.array([
            [0, 0, 0],
            [s, 0, 0],
            [s, s, 0],
            [0, s, 0],
        ], dtype=np.float32)
        all_obj_pts.append(obj)
        all_img_pts.append(corner.reshape(4, 1, 2))

    print(f"  ok ({len(ids)} tags): {p.name}")

print(f"\ncalibrating with {len(all_obj_pts)} detections...")
rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    all_obj_pts, all_img_pts, img_size, None, None
)
print(f"rms: {rms:.4f} px")
print(f"camera matrix:\n{camera_matrix}")
print(f"distortion: {dist_coeffs.ravel()}")

np.savez(CALIB_OUT,
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs,
         rms=rms)
print(f"\nsaved -> {CALIB_OUT}")
