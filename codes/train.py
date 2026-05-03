from pathlib import Path
from collections import Counter
import random
import math
import numpy as np
from PIL import Image, ImageFile
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import timm
from timm.data import Mixup, create_transform
from timm.scheduler import CosineLRScheduler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# paths
SEED        = 42
PROJECT_DIR = Path("E:/Computer Vision")
ROOT        = PROJECT_DIR / "dataset" / "khana"
SAVE_PATH   = PROJECT_DIR / "checkpoints" / "khana_best.pt"

# model - pretrained on imagenet21k then finetuned on in1k, much better than in1k only
ARCH           = "convnext_base.fb_in22k_ft_in1k"
DROP_PATH_RATE = 0.2   # stochastic depth for regularization

# training
BATCH_SIZE   = 32       # can try 64 if enough vram
NUM_WORKERS  = 4
WEIGHT_DECAY = 0.05     # convnext needs higher weight decay than usual
GRAD_CLIP    = 1.0
LABEL_SMOOTH = 0.1

# stage 1 - larger lr, 224px, most of the training happens here
EPOCHS_S1  = 20
IMG_S1     = 224
LR_S1      = 3e-4
LR_MIN_S1  = 1e-6
WARMUP_S1  = 5    # warmup epochs before cosine decay starts

# stage 2 - smaller lr, 320px finetune
EPOCHS_S2  = 8
IMG_S2     = 320
LR_S2      = 5e-5   # very small since just finetuning at higher res
LR_MIN_S2  = 1e-7
WARMUP_S2  = 1

# mixup / cutmix params
MIXUP_ALPHA   = 0.4
CUTMIX_ALPHA  = 1.0
MIXUP_PROB    = 1.0
CUTMIX_MINMAX = None
SWITCH_PROB   = 0.5   # 50% chance of switching between cutmix and mixup

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# data
if not ROOT.exists():
    raise FileNotFoundError("dataset not found: " + str(ROOT))

class_dirs   = sorted([p for p in ROOT.iterdir() if p.is_dir()])
classes      = [p.name for p in class_dirs]
class_to_idx = {name: i for i, name in enumerate(classes)}

samples = []
for class_dir in class_dirs:
    label = class_to_idx[class_dir.name]
    for p in class_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(("._", ".")):
            continue
        samples.append((str(p), label))

if not samples:
    raise RuntimeError("no images found")

labels = [y for _, y in samples]
paths  = [p for p, _ in samples]
counts = Counter(labels)
print(f"classes: {len(classes)} | images: {len(samples)}")
print(f"min/median/max per class: {min(counts.values())} / {int(np.median(list(counts.values())))} / {max(counts.values())}")

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths, labels, test_size=0.2, random_state=SEED, stratify=labels
)
print(f"train: {len(train_paths)} | val: {len(val_paths)}")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def make_train_tf(img_size):
    # randaugment + random erasing, stronger than manual augmentations
    return create_transform(
        input_size        = img_size,
        is_training       = True,
        auto_augment      = "rand-m9-mstd0.5-inc1",
        re_prob           = 0.25,    # random erasing probability
        re_mode           = "pixel",
        re_count          = 1,
        interpolation     = "bicubic",
        mean              = IMAGENET_MEAN,
        std               = IMAGENET_STD,
        scale             = (0.65, 1.0),
        ratio             = (3.0/4.0, 4.0/3.0),
        hflip             = 0.5,
        vflip             = 0.0,
        color_jitter      = 0.4,
        color_jitter_prob = 0.8,
    )


def make_val_tf(img_size):
    return T.Compose([
        T.Resize(int(img_size * 1.143), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class KhanaDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.items     = list(zip(paths, labels))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        for attempt in range(5):
            path, label = self.items[(idx + attempt) % len(self.items)]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
            except Exception:
                continue
        raise RuntimeError("couldn't load image at index " + str(idx))


# mixup + cutmix setup - produces soft targets so normal CE won't work
mixup_fn = Mixup(
    mixup_alpha    = MIXUP_ALPHA,
    cutmix_alpha   = CUTMIX_ALPHA,
    cutmix_minmax  = CUTMIX_MINMAX,
    prob           = MIXUP_PROB,
    switch_prob    = SWITCH_PROB,
    mode           = "batch",
    label_smoothing= LABEL_SMOOTH,
    num_classes    = len(classes),
)


class SoftTargetCrossEntropy(nn.Module):
    # regular CE doesn't work with soft labels from mixup
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()


# model
def build_model(num_classes: int) -> nn.Module:
    model = timm.create_model(
        ARCH,
        pretrained     = True,
        num_classes    = num_classes,
        drop_path_rate = DROP_PATH_RATE,
    )
    return model


# layer-wise lr decay - backbone gets lower lr, head gets full lr
def build_param_groups(model: nn.Module, base_lr: float, decay: float = 0.75):
    param_groups = []
    # head gets base lr
    param_groups.append({
        "params": list(model.head.parameters()),
        "lr": base_lr,
        "name": "head",
    })
    # each earlier stage gets multiplied by decay
    stages = [model.stages[i] for i in range(len(model.stages) - 1, -1, -1)]
    for i, stage in enumerate(stages):
        param_groups.append({
            "params": list(stage.parameters()),
            "lr": base_lr * (decay ** (i + 1)),
            "name": f"stage{len(stages) - 1 - i}",
        })
    stem_params = list(model.stem.parameters())
    if hasattr(model, "norm_pre"):
        stem_params += list(model.norm_pre.parameters())
    param_groups.append({
        "params": stem_params,
        "lr": base_lr * (decay ** (len(stages) + 1)),
        "name": "stem",
    })
    for g in param_groups:
        print(f"  {g['name']:10s}  lr={g['lr']:.2e}")
    return param_groups


@torch.no_grad()
def evaluate(model, loader, tta=False):
    model.eval()
    correct_1 = correct_5 = total = 0

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if tta:
            # 5-crop: 4 corners + center, average the logits
            bs, c, h, w = x.shape
            crop_h = crop_w = int(h * 0.875)
            crops = [
                x[:, :, :crop_h,  :crop_w ],
                x[:, :, :crop_h,  -crop_w:],
                x[:, :, -crop_h:, :crop_w ],
                x[:, :, -crop_h:, -crop_w:],
                x[:, :,
                  (h - crop_h) // 2:(h + crop_h) // 2,
                  (w - crop_w) // 2:(w + crop_w) // 2],
            ]
            logits = sum(
                model(crop.to(device))
                for crop in crops
            ) / len(crops)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
                logits = model(x)

        pred1 = logits.argmax(dim=1)
        top5  = logits.topk(min(5, logits.size(1)), dim=1).indices
        correct_1 += (pred1 == y).sum().item()
        correct_5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.size(0)

    return correct_1 / total, correct_5 / total


def run_stage(
    model, train_paths, train_labels, val_paths, val_labels,
    img_size, epochs, base_lr, lr_min, warmup_epochs,
    best_acc=0.0, stage_name="S1"
):
    print(f"\n{'='*60}")
    print(f"  {stage_name}  img={img_size}  epochs={epochs}  lr={base_lr:.1e}")
    print(f"{'='*60}")

    train_ds = KhanaDataset(train_paths, train_labels, transform=make_train_tf(img_size))
    val_ds   = KhanaDataset(val_paths,   val_labels,   transform=make_val_tf(img_size))

    # weighted sampler so each class is seen equally
    train_class_counts = Counter(train_labels)
    sample_weights = [1.0 / train_class_counts[y] for y in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    print("lr groups:")
    param_groups = build_param_groups(model, base_lr, decay=0.75)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    # cosine lr with linear warmup
    steps_per_epoch = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial      = epochs * steps_per_epoch,
        lr_min         = lr_min,
        warmup_t       = warmup_epochs * steps_per_epoch,
        warmup_lr_init = 1e-7,
        warmup_prefix  = True,
        cycle_limit    = 1,
        t_in_epochs    = False,   # step-level updates not epoch-level
    )

    criterion = SoftTargetCrossEntropy()
    scaler    = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"{stage_name} ep{epoch+1}")):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x, y_soft = mixup_fn(x, y)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
                logits = model(x)
                loss   = criterion(logits, y_soft)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(epoch * steps_per_epoch + step + 1)

            running_loss += loss.item()

        val_acc1, val_acc5 = evaluate(model, val_loader, tta=False)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{stage_name} ep{epoch+1:02d} | "
            f"loss {running_loss/steps_per_epoch:.4f} | "
            f"val@1 {val_acc1:.4f} | val@5 {val_acc5:.4f} | "
            f"lr {cur_lr:.2e}"
        )

        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save(
                {
                    "model_state":  model.state_dict(),
                    "classes":      classes,
                    "class_to_idx": class_to_idx,
                    "img_size":     img_size,
                    "arch":         ARCH,
                    "val_acc1":     best_acc,
                },
                SAVE_PATH,
            )
            print(f"  saved best ({best_acc:.4f}) -> {SAVE_PATH}")

    return best_acc


if __name__ == "__main__":
    num_classes = len(classes)
    print(f"\nbuilding {ARCH} | classes={num_classes}")
    model = build_model(num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {total_params:.1f}M")

    # stage 1 - 224px
    best = run_stage(
        model, train_paths, train_labels, val_paths, val_labels,
        img_size     = IMG_S1,
        epochs       = EPOCHS_S1,
        base_lr      = LR_S1,
        lr_min       = LR_MIN_S1,
        warmup_epochs= WARMUP_S1,
        best_acc     = 0.0,
        stage_name   = "S1",
    )

    # stage 2 - load best weights then finetune at 320px
    print("\nloading best s1 weights...")
    ckpt = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    best = run_stage(
        model, train_paths, train_labels, val_paths, val_labels,
        img_size     = IMG_S2,
        epochs       = EPOCHS_S2,
        base_lr      = LR_S2,
        lr_min       = LR_MIN_S2,
        warmup_epochs= WARMUP_S2,
        best_acc     = best,
        stage_name   = "S2",
    )

    # tta eval on best checkpoint
    print("\nrunning tta on best checkpoint...")
    ckpt = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_ds_tta = KhanaDataset(val_paths, val_labels, transform=make_val_tf(IMG_S2))
    val_loader_tta = DataLoader(
        val_ds_tta, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    tta_acc1, tta_acc5 = evaluate(model, val_loader_tta, tta=True)
    print(f"\nfinal tta val@1: {tta_acc1:.4f} | val@5: {tta_acc5:.4f}")
    print(f"best val@1 (no tta): {best:.4f}")
