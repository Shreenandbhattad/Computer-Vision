from pathlib import Path
import csv
import torch
from PIL import Image, ImageFile
import torchvision.transforms as T
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_DIR = Path("E:/Computer Vision")
CKPT_PATH   = PROJECT_DIR / "checkpoints" / "khana_best.pt"
IMAGE_DIR   = PROJECT_DIR / "test_images"
OUT_CSV     = PROJECT_DIR / "predictions.csv"

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

if not CKPT_PATH.exists():
    raise FileNotFoundError("checkpoint not found: " + str(CKPT_PATH))

if not IMAGE_DIR.exists():
    raise FileNotFoundError("image folder not found: " + str(IMAGE_DIR))

# load checkpoint
ckpt = torch.load(CKPT_PATH, map_location=device)
classes  = ckpt["classes"]
img_size = ckpt.get("img_size", 320)
arch     = ckpt.get("arch", "convnext_base.fb_in22k_ft_in1k")

# model
model = timm.create_model(arch, pretrained=False, num_classes=len(classes))
model.load_state_dict(ckpt["model_state"])
model = model.to(device)
model.eval()

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

tf = T.Compose([
    T.Resize(int(img_size * 1.143), interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(img_size),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def load_image(path: Path):
    with Image.open(path) as img:
        return img.convert("RGB")

# collect images
image_paths = []
for p in sorted(IMAGE_DIR.rglob("*")):
    if p.is_file() and p.suffix.lower() in img_exts and not p.name.startswith("._"):
        image_paths.append(p)

if not image_paths:
    raise RuntimeError("no images found in " + str(IMAGE_DIR))

rows = []

with torch.no_grad():
    for path in image_paths:
        img = load_image(path)
        x   = tf(img).unsqueeze(0).to(device)

        logits = model(x)
        prob   = torch.softmax(logits, dim=1)[0]
        top5   = torch.topk(prob, k=min(5, len(classes)))

        pred_idx   = int(top5.indices[0].item())
        pred_class = classes[pred_idx]
        pred_conf  = float(top5.values[0].item())

        top5_text = "; ".join(
            f"{classes[idx]}:{score:.4f}"
            for score, idx in zip(top5.values.tolist(), top5.indices.tolist())
        )

        rows.append([path.name, pred_class, pred_conf, top5_text])

        print(path.name)
        print("  prediction:", pred_class)
        print("  confidence:", f"{pred_conf:.4f}")
        print("  top5:", top5_text)
        print()

# save results
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "prediction", "confidence", "top5"])
    writer.writerows(rows)

print("saved to", OUT_CSV)
