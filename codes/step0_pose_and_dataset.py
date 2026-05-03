from pathlib import Path
import time
import cv2
import numpy as np

PROJECT_DIR = Path("E:/Computer Vision")

# config
CALIB_NPZ   = PROJECT_DIR / "nerf_data" / "camera_calib.npz"
SCAN_DIR    = PROJECT_DIR / "nerf_data" / "thali_images"
OUT_NPZ     = PROJECT_DIR / "nerf_data" / "thali_dataset.npz"
OUT_IMG_DIR = PROJECT_DIR / "nerf_data" / "undistorted"

TAG_SIZE_M = 0.025   # same value as used in calibration
IMG_RESIZE = 400     # resize images to this width (None = keep original)

TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1

# load calibration
calib = np.load(CALIB_NPZ)
K     = calib["camera_matrix"]
D     = calib["dist_coeffs"]
print(f"loaded calibration. K =\n{K}")

aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# tag 3d corners on Z=0 plane
s = TAG_SIZE_M
tag_3d = np.array([
    [0, 0, 0],
    [s, 0, 0],
    [s, s, 0],
    [0, s, 0],
], dtype=np.float32)

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

img_paths = sorted(SCAN_DIR.glob("*"))
img_paths = [p for p in img_paths if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
print(f"\nfound {len(img_paths)} scan images")

images_list = []
c2ws_list   = []
skipped     = 0

for p in img_paths:
    img_bgr = cv2.imread(str(p))
    if img_bgr is None:
        skipped += 1; continue

    h_orig, w_orig = img_bgr.shape[:2]

    # undistort
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w_orig, h_orig), alpha=0)
    undist = cv2.undistort(img_bgr, K, D, None, new_K)
    x_roi, y_roi, w_roi, h_roi = roi
    undist = undist[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    K_crop = new_K.copy()
    K_crop[0, 2] -= x_roi
    K_crop[1, 2] -= y_roi

    # optional resize
    if IMG_RESIZE is not None:
        scale  = IMG_RESIZE / undist.shape[1]
        undist = cv2.resize(undist, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
        K_crop = K_crop * scale
        K_crop[2, 2] = 1.0

    # detect aruco in undistorted image
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is None or len(ids) == 0:
        print(f"  skip (no tag): {p.name}")
        skipped += 1; continue

    img_pts = corners[0].reshape(4, 1, 2)

    # pnp to get camera pose
    success, rvec, tvec = cv2.solvePnP(
        tag_3d, img_pts, K_crop, None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not success:
        print(f"  skip (pnp failed): {p.name}")
        skipped += 1; continue

    # world-to-camera to camera-to-world
    R, _ = cv2.Rodrigues(rvec)
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3,  3] = -(R.T @ tvec).ravel()

    # opencv to opengl coords (flip y and z)
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1

    img_rgb = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
    images_list.append(img_rgb)
    c2ws_list.append(c2w)

    out_p = OUT_IMG_DIR / p.name
    cv2.imwrite(str(out_p), undist)
    print(f"  ok: {p.name}  t={c2w[:3,3].round(3)}")

print(f"\nsuccessful: {len(images_list)} | skipped: {skipped}")

if len(images_list) < 5:
    raise RuntimeError("not enough images, make sure aruco tag is clearly visible")

# viser visualization
try:
    import viser
    server = viser.ViserServer(share=True)
    H_vis, W_vis = images_list[0].shape[:2]
    focal_vis = float(K_crop[0, 0])
    for i, (img, c2w) in enumerate(zip(images_list, c2ws_list)):
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov      = 2 * np.arctan2(H_vis / 2, focal_vis),
            aspect   = W_vis / H_vis,
            scale    = 0.02,
            wxyz     = viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position = c2w[:3, 3],
            image    = img,
        )
    print("\nviser running - open the url in a browser.")
    print("take 2 screenshots then press ctrl-c to continue.\n")
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("viser closed. saving dataset...")
except ImportError:
    print("viser not installed, skipping. pip install viser")

# train/val/test split
N = len(images_list)
indices = np.random.permutation(N)
n_train = int(N * TRAIN_FRAC)
n_val   = int(N * VAL_FRAC)

tr = indices[:n_train]
va = indices[n_train:n_train+n_val]
te = indices[n_train+n_val:]

images_arr = np.stack(images_list, axis=0).astype(np.uint8)
c2ws_arr   = np.stack(c2ws_list,   axis=0).astype(np.float32)

focal = float(K_crop[0, 0])

np.savez(
    OUT_NPZ,
    images_train = images_arr[tr],
    c2ws_train   = c2ws_arr[tr],
    images_val   = images_arr[va],
    c2ws_val     = c2ws_arr[va],
    c2ws_test    = c2ws_arr[te],
    focal        = focal,
)
print(f"\ndataset saved -> {OUT_NPZ}")
print(f"  train={len(tr)}  val={len(va)}  test={len(te)}")
print(f"  image shape: {images_arr[0].shape}")
print(f"  focal: {focal:.2f} px")
