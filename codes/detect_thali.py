from pathlib import Path
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# paths
PROJECT_DIR = Path("E:/Computer Vision")
CKPT_PATH   = PROJECT_DIR / "checkpoints" / "khana_best.pt"
SAM_CKPT    = PROJECT_DIR / "checkpoints" / "sam_vit_h_4b8939-002.pth"
SAM_MODEL   = "vit_h"   # vit_b is faster but less accurate

# detection thresholds
CONF_THRESH   = 0.75   # min confidence to keep a detection
IOU_THRESH    = 0.25   # nms overlap threshold
MIN_AREA_FRAC = 0.010  # each bowl should be at least ~1% of plate area
MAX_AREA_FRAC = 0.12  
EXCLUDE_LABELS = {"thali", "plate"} # classes to skip  not food items
MIN_ASPECT = 0.20   # min width/height ratio
MAX_ASPECT = 5.0    # max width/height ratio
CROP_PAD   = 10     # padding around each mask bbox before classifying
SAM_PTS_PER_SIDE   = 48
SAM_IOU_THRESH     = 0.88
SAM_STAB_THRESH    = 0.95   # higher = cleaner bowl boundaries
SAM_BOX_NMS_THRESH = 0.70

# classifier transform
IMG_SIZE      = 320
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# classifier
def load_classifier(ckpt_path: Path):
    ckpt    = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    arch    = ckpt.get("arch", "convnext_base.fb_in22k_ft_in1k")
    model   = timm.create_model(arch, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    print(f"classifier: {arch} | {len(classes)} classes")
    return model, classes


clf_tf = T.Compose([
    T.Resize(int(IMG_SIZE * 1.143), interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@torch.no_grad()
def classify_crop(model, classes, pil_crop: Image.Image):
    x      = clf_tf(pil_crop).unsqueeze(0).to(device)
    logits = model(x)
    prob   = torch.softmax(logits, dim=1)[0]
    top5   = torch.topk(prob, k=min(5, len(classes)))
    label  = classes[int(top5.indices[0])]
    conf   = float(top5.values[0])
    top5_l = [(classes[int(i)], float(v)) for v, i in zip(top5.values, top5.indices)]
    return label, conf, top5_l


# sam
def load_sam(sam_ckpt: Path, model_type: str):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[model_type](checkpoint=str(sam_ckpt))
    sam.to(device)
    mg = SamAutomaticMaskGenerator(
        sam,
        points_per_side        = SAM_PTS_PER_SIDE,
        pred_iou_thresh        = SAM_IOU_THRESH,
        stability_score_thresh = SAM_STAB_THRESH,
        box_nms_thresh         = SAM_BOX_NMS_THRESH,
    )
    print("sam loaded:", model_type)
    return mg


def bev_interactive(img_bgr: np.ndarray) -> np.ndarray:
    print("\n[BEV]  4 plate corners: TL -> TR -> BR -> BL")
    print("      ENTER to warp | ESC to skip\n")

    pts   = []
    clone = img_bgr.copy()

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(clone, (x, y), 8, (0, 255, 80), -1)
            if len(pts) > 1:
                cv2.line(clone, pts[-2], pts[-1], (0, 255, 80), 2)
            if len(pts) == 4:
                cv2.line(clone, pts[-1], pts[0], (0, 255, 80), 2)
            cv2.imshow("BEV", clone)

    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("BEV", cb)
    cv2.imshow("BEV", clone)

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows(); return img_bgr
        if k == 13 and len(pts) == 4:
            break
    cv2.destroyAllWindows()

    h, w  = img_bgr.shape[:2]
    side  = min(h, w)
    src   = np.float32(pts)
    dst   = np.float32([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]])
    M     = cv2.getPerspectiveTransform(src, dst)
    warp  = cv2.warpPerspective(img_bgr, M, (side, side), flags=cv2.INTER_CUBIC)
    print(f"[BEV] warp done -> {side}x{side}")
    return warp


#  bev auto  hough circle to find plate then warp
def bev_auto(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist   = min(h, w) // 3,
        param1    = 100, param2 = 40,
        minRadius = min(h, w) // 8,
        maxRadius = int(min(h, w) * 0.55),
    )

    if circles is None:
        print("[BEV] no circle found, using original image")
        return img_bgr

    xc, yc, r = map(int, circles[0][0])
    print(f"[BEV] circle at ({xc},{yc}) r={r}")

    src = np.float32([
        [xc - r, yc - r],
        [xc + r, yc - r],
        [xc + r, yc + r],
        [xc - r, yc + r],
    ])
    side = int(2 * r)
    dst  = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    M    = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img_bgr, M, (side, side), flags=cv2.INTER_CUBIC)
    print(f"[BEV] auto warp -> {side}x{side}")
    return warp


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua    = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets, iou_thr):
    # per-class nms first then global
    from collections import defaultdict
    by_class = defaultdict(list)
    for d in dets:
        by_class[d["label"]].append(d)

    merged = []
    for label, group in by_class.items():
        group = sorted(group, key=lambda d: d["conf"], reverse=True)
        keep = []
        while group:
            best = group.pop(0)
            keep.append(best)
            group = [d for d in group if iou(best["box"], d["box"]) < iou_thr]
        merged.extend(keep)

    # global nms across all classes
    merged = sorted(merged, key=lambda d: d["conf"], reverse=True)
    final = []
    while merged:
        best = merged.pop(0)
        final.append(best)
        merged = [d for d in merged if iou(best["box"], d["box"]) < iou_thr]
    return final


PALETTE = [
    (255,  87,  34), (33, 150, 243), (76, 175,  80),
    (255, 193,   7), (156, 39, 176), (0,  188, 212),
    (255,  87,  87), (0,  230, 118), (255, 235,  59),
    (121,  85,  72), (233,  30,  99), (63, 81, 181),
]


def detect(img_path, model, classes, mask_gen, bev_mode="none", out_path=None):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError("could not load image: " + str(img_path))
    print(f"\nimage: {img_path}  {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    # bev warp if needed
    if bev_mode == "interactive":
        img_bgr = bev_interactive(img_bgr)
    elif bev_mode == "auto":
        img_bgr = bev_auto(img_bgr)

    h, w     = img_bgr.shape[:2]
    img_area = h * w

    if bev_mode in ("auto", "interactive"):
        bev_out = str(Path(img_path).parent / (Path(img_path).stem + "_bev.jpg"))
        cv2.imwrite(bev_out, img_bgr)
        print(f"bev saved -> {bev_out}")

    # sam masks
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print("running sam...")
    masks = mask_gen.generate(img_rgb)
    print(f"sam: {len(masks)} raw masks")

    # classify each crop
    pil_img    = Image.fromarray(img_rgb)
    detections = []

    for m in masks:
        bx, by, bw, bh = [int(v) for v in m["bbox"]]
        area_frac = m["area"] / img_area

        if area_frac < MIN_AREA_FRAC or area_frac > MAX_AREA_FRAC:
            continue
        if bh == 0 or not (MIN_ASPECT <= bw / bh <= MAX_ASPECT):
            continue

        cx1 = max(0, bx - CROP_PAD);  cy1 = max(0, by - CROP_PAD)
        cx2 = min(w, bx + bw + CROP_PAD); cy2 = min(h, by + bh + CROP_PAD)
        crop = pil_img.crop((cx1, cy1, cx2, cy2))

        if crop.size[0] < 24 or crop.size[1] < 24:
            continue

        label, conf, top5 = classify_crop(model, classes, crop)
        if conf < CONF_THRESH:
            continue
        if label in EXCLUDE_LABELS:
            continue

        detections.append({
            "box":   [bx, by, bx + bw, by + bh],
            "label": label,
            "conf":  conf,
            "top5":  top5,
        })

    print(f"before nms: {len(detections)}  |  after nms: ", end="")
    detections = nms(detections, IOU_THRESH)
    print(len(detections))

    # draw boxes
    lbl_color = {}
    for d in detections:
        if d["label"] not in lbl_color:
            lbl_color[d["label"]] = PALETTE[len(lbl_color) % len(PALETTE)]

    out_img = img_bgr.copy()
    font    = cv2.FONT_HERSHEY_SIMPLEX

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = lbl_color[d["label"]]
        text  = f"{d['label']}  {d['conf']:.2f}"

        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        cv2.rectangle(out_img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out_img, text, (x1 + 3, y1 - 4), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # save output
    if out_path is None:
        p = Path(img_path)
        out_path = str(p.parent / f"{p.stem}_detected.jpg")

    cv2.imwrite(out_path, out_img)
    print(f"saved -> {out_path}")

    # summary
    print(f"  {'#':>3}  {'Label':30s}  {'Conf':>5}  Box")
    for i, d in enumerate(detections, 1):
        x1,y1,x2,y2 = d["box"]
        print(f"  {i:3d}  {d['label']:30s}  {d['conf']:.3f}  [{x1},{y1},{x2},{y2}]")
    print(f"{'─'*55}")

    return detections, out_img


def parse_args():
    p = argparse.ArgumentParser(description="Khana Thali Detector")
    p.add_argument("image",        help="input image path")
    p.add_argument("--out",        default=None,  help="output image path")
    p.add_argument("--bev",        default="none",
                   choices=["none", "auto", "interactive"],
                   help="bev mode")
    p.add_argument("--ckpt",       default=str(CKPT_PATH))
    p.add_argument("--sam",        default=str(SAM_CKPT))
    p.add_argument("--sam-model",  default=SAM_MODEL,
                   choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--conf",       type=float, default=CONF_THRESH)
    p.add_argument("--iou",        type=float, default=IOU_THRESH)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONF_THRESH = args.conf
    IOU_THRESH  = args.iou

    print("loading classifier...")
    model, classes = load_classifier(Path(args.ckpt))

    print("loading sam...")
    mask_gen = load_sam(Path(args.sam), args.sam_model)

    detect(
        img_path = args.image,
        model    = model,
        classes  = classes,
        mask_gen = mask_gen,
        bev_mode = args.bev,
        out_path = args.out,
    )
