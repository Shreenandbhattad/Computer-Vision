from pathlib import Path
from collections import Counter
import argparse, copy, random
import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
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
SAVE_PATH   = PROJECT_DIR / "checkpoints" / "khana_best_v2.pt"
SAVE_EMA    = PROJECT_DIR / "checkpoints" / "khana_best_v2_ema.pt"

# model - eva02 large, much stronger than convnext, native 448px
ARCH           = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
DROP_PATH_RATE = 0.3   # higher than v1, larger model needs more regularization

# cosine classifier - l2 normalize features + weights, sharper clusters
USE_COSINE_CLF = True
TEMPERATURE    = 0.07   # lower = sharper, controls how peaked the output is

# stage configs: img_size, epochs, lr, min_lr, warmup_epochs
S1_IMG = 224;  S1_EP = 15;  S1_LR = 2e-4;  S1_LRMIN = 1e-6;  S1_WU = 3
S2_IMG = 320;  S2_EP = 8;   S2_LR = 4e-5;  S2_LRMIN = 1e-7;  S2_WU = 1
S3_IMG = 448;  S3_EP = 6;   S3_LR = 1e-5;  S3_LRMIN = 5e-8;  S3_WU = 1

# training
BATCH_SIZE   = 16    # lower than v1, 448px needs more memory
NUM_WORKERS  = 4
WEIGHT_DECAY = 0.05
GRAD_CLIP    = 1.0
LABEL_SMOOTH = 0.1
MIXUP_ALPHA  = 0.4
CUTMIX_ALPHA = 1.0
MIXUP_PROB   = 1.0
SWITCH_PROB  = 0.5

# focal loss - focuses on hard examples, gamma=2 is standard
USE_FOCAL   = True
FOCAL_GAMMA = 2.0

# ema - exponential moving average of weights, usually gives free accuracy
EMA_DECAY = 0.9998   # high value means slow moving average (~5000 step memory)

# swa
USE_SWA = True
SWA_LR  = 5e-6

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# data
class_dirs   = sorted([p for p in ROOT.iterdir() if p.is_dir()])
classes      = [p.name for p in class_dirs]
class_to_idx = {n: i for i, n in enumerate(classes)}

samples = []
for d in class_dirs:
    lbl = class_to_idx[d.name]
    for p in d.rglob("*"):
        if not p.is_file(): continue
        if p.name.startswith(("._", ".")): continue
        if p.suffix.lower() not in {".jpg",".jpeg",".png",".webp",".bmp"}: continue
        samples.append((str(p), lbl))

labels = [y for _,y in samples]
paths  = [p for p,_ in samples]
counts = Counter(labels)
print(f"classes: {len(classes)} | images: {len(samples)}")
print(f"min/med/max: {min(counts.values())} / {int(np.median(list(counts.values())))} / {max(counts.values())}")

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths, labels, test_size=0.2, random_state=SEED, stratify=labels)
print(f"train: {len(train_paths)} | val: {len(val_paths)}")


def make_train_tf(img_size):
    return create_transform(
        input_size        = img_size,
        is_training       = True,
        auto_augment      = "rand-m9-mstd0.5-inc1",
        re_prob           = 0.25,
        re_mode           = "pixel",
        interpolation     = "bicubic",
        mean              = MEAN, std = STD,
        scale             = (0.60, 1.0),
        ratio             = (3/4, 4/3),
        hflip             = 0.5,
        color_jitter      = 0.4,
        color_jitter_prob = 0.8,
    )

def make_val_tf(img_size):
    return T.Compose([
        T.Resize(int(img_size * 1.143), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


class KhanaDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.items = list(zip(paths, labels))
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        for k in range(5):
            path, label = self.items[(idx+k) % len(self.items)]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform: img = self.transform(img)
                return img, label
            except Exception: continue
        raise RuntimeError("couldn't load image at " + str(idx))


# cosine classifier - angle between features matters, not magnitude
class CosineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x,          dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return x_norm @ w_norm.T / self.temperature


# focal loss - down-weights easy examples so model focuses on hard pairs
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = log_probs.exp()
        focal_w   = (1.0 - probs) ** self.gamma
        loss = -(targets * focal_w * log_probs).sum(dim=-1).mean()
        return loss


# ema - keeps a smoothed copy of weights, usually gives +0.3-0.5% for free
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(m)

    def state_dict(self): return self.shadow.state_dict()


# layer-wise lr decay - head gets full lr, earlier layers get less
def build_param_groups(model, base_lr, decay=0.80):
    no_decay = {"bias", "norm", "gamma", "beta"}
    param_groups = []

    head_params = [p for n,p in model.named_parameters()
                   if ("head" in n or "classifier" in n) and p.requires_grad]
    param_groups.append({"params": head_params, "lr": base_lr, "name": "head",
                          "weight_decay": 0.0})

    block_names = []
    if hasattr(model, "blocks"):
        block_names = [f"blocks.{i}" for i in range(len(model.blocks))]
    elif hasattr(model, "stages"):
        block_names = [f"stages.{i}" for i in range(len(model.stages))]

    for depth, bname in enumerate(reversed(block_names)):
        params = [p for n,p in model.named_parameters()
                  if bname in n and p.requires_grad
                  and not any(nd in n for nd in no_decay)]
        params_nd = [p for n,p in model.named_parameters()
                     if bname in n and p.requires_grad
                     and any(nd in n for nd in no_decay)]
        lr_i = base_lr * (decay ** (depth + 1))
        if params:
            param_groups.append({"params": params,    "lr": lr_i, "name": bname,
                                  "weight_decay": WEIGHT_DECAY})
        if params_nd:
            param_groups.append({"params": params_nd, "lr": lr_i, "name": bname+"_nd",
                                  "weight_decay": 0.0})

    stem_params = [p for n,p in model.named_parameters()
                   if ("patch_embed" in n or "stem" in n) and p.requires_grad]
    if stem_params:
        param_groups.append({"params": stem_params,
                              "lr": base_lr * (decay ** (len(block_names)+1)),
                              "name": "stem", "weight_decay": WEIGHT_DECAY})

    for g in param_groups[:5]:
        print(f"  {g['name']:30s}  lr={g['lr']:.2e}")
    print(f"  ... ({len(param_groups)} groups total)")
    return param_groups


@torch.no_grad()
def evaluate(model, loader, tta=False, img_size=320):
    model.eval()
    correct_1 = correct_5 = total = 0

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if tta:
            # 20-view tta: 3 scales x (5 crops + hflip each) = 30 but we get ~20 unique
            bs, c, h, w = x.shape
            views = []
            for scale in [1.0, 0.875, 0.75]:
                ch = int(h * scale); cw = int(w * scale)
                crops = [
                    x[:, :, :ch,      :cw     ],
                    x[:, :, :ch,      -cw:    ],
                    x[:, :, -ch:,     :cw     ],
                    x[:, :, -ch:,     -cw:    ],
                    x[:, :, (h-ch)//2:(h+ch)//2, (w-cw)//2:(w+cw)//2],
                ]
                for crop in crops:
                    if crop.shape[-2:] != (h, w):
                        crop = F.interpolate(crop, size=(h,w), mode="bilinear",
                                             align_corners=False)
                    views.append(crop)
                    views.append(torch.flip(crop, dims=[-1]))
            logits = sum(model(v) for v in views) / len(views)
        else:
            with torch.autocast("cuda", torch.float16, enabled=(device=="cuda")):
                logits = model(x)

        pred1 = logits.argmax(1)
        top5  = logits.topk(min(5, logits.size(1)), 1).indices
        correct_1 += (pred1 == y).sum().item()
        correct_5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
        total     += y.size(0)

    return correct_1 / total, correct_5 / total


# model
def build_model(num_classes):
    model = timm.create_model(
        ARCH,
        pretrained     = True,
        num_classes    = 0,   # remove default head, we attach our own
        drop_path_rate = DROP_PATH_RATE,
    )
    with torch.no_grad():
        dummy    = torch.zeros(1, 3, 224, 224)
        feat_dim = model(dummy).shape[-1]

    if USE_COSINE_CLF:
        model.head = CosineClassifier(feat_dim, num_classes, TEMPERATURE)
    else:
        model.head = nn.Linear(feat_dim, num_classes)

    return model


mixup_fn = Mixup(
    mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA,
    prob=MIXUP_PROB, switch_prob=SWITCH_PROB,
    mode="batch", label_smoothing=LABEL_SMOOTH,
    num_classes=len(classes),
)

criterion = FocalLoss(gamma=FOCAL_GAMMA) if USE_FOCAL else \
            (lambda logits, targets: -(targets * F.log_softmax(logits,-1)).sum(-1).mean())


def run_stage(model, ema, img_size, epochs, base_lr, lr_min, warmup_ep,
              best_acc=0.0, tag="S1", use_swa=False):
    print(f"\n{'='*60}\n  {tag}  img={img_size}  epochs={epochs}  lr={base_lr:.1e}\n{'='*60}")

    train_ds = KhanaDataset(train_paths, train_labels, make_train_tf(img_size))
    val_ds   = KhanaDataset(val_paths,   val_labels,   make_val_tf(img_size))

    w = [1.0 / Counter(train_labels)[y] for y in train_labels]
    sampler = WeightedRandomSampler(w, len(w), replacement=True)

    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)

    print("lr groups:")
    param_groups = build_param_groups(model, base_lr, decay=0.80)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    steps = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=epochs*steps, lr_min=lr_min,
        warmup_t=warmup_ep*steps, warmup_lr_init=1e-8,
        warmup_prefix=True, t_in_epochs=False,
    )

    # swa setup
    if use_swa and USE_SWA:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_sched = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=3)
        swa_start = max(1, epochs // 2)
        print(f"swa starts at epoch {swa_start}")
    else:
        swa_model = None

    scaler = torch.amp.GradScaler("cuda", enabled=(device=="cuda"))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"{tag} ep{epoch+1}")):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x, y_soft = mixup_fn(x, y)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=(device=="cuda")):
                logits = model(x)
                loss   = criterion(logits, y_soft)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()

            ema.update(model)
            scheduler.step_update(epoch * steps + step + 1)
            running_loss += loss.item()

        if swa_model and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()

        val_acc1, val_acc5 = evaluate(model, val_loader)
        ema_acc1, _        = evaluate(ema.shadow, val_loader)
        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"{tag} ep{epoch+1:02d} | "
              f"loss {running_loss/steps:.4f} | "
              f"val@1 {val_acc1:.4f} | ema@1 {ema_acc1:.4f} | "
              f"val@5 {val_acc5:.4f} | lr {cur_lr:.2e}")

        # prefer ema weights when saving
        best_model = ema.shadow if ema_acc1 > val_acc1 else model
        best_score = max(ema_acc1, val_acc1)
        if best_score > best_acc:
            best_acc = best_score
            torch.save({
                "model_state":  best_model.state_dict(),
                "classes":      classes,
                "class_to_idx": class_to_idx,
                "img_size":     img_size,
                "arch":         ARCH,
                "val_acc1":     best_acc,
                "cosine_clf":   USE_COSINE_CLF,
            }, SAVE_PATH)
            print(f"  saved best ({best_acc:.4f}) -> {SAVE_PATH}")

    if swa_model and USE_SWA:
        print("updating swa bn stats...")
        swa_bn_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=True)
        torch.optim.swa_utils.update_bn(swa_bn_loader, swa_model, device=device)
        swa_acc1, _ = evaluate(swa_model, val_loader)
        print(f"swa val@1: {swa_acc1:.4f}")
        if swa_acc1 > best_acc:
            best_acc = swa_acc1
            torch.save({
                "model_state":  swa_model.module.state_dict(),
                "classes":      classes,
                "class_to_idx": class_to_idx,
                "img_size":     img_size,
                "arch":         ARCH,
                "val_acc1":     best_acc,
                "cosine_clf":   USE_COSINE_CLF,
            }, SAVE_PATH)
            print(f"  saved swa best ({best_acc:.4f}) -> {SAVE_PATH}")

    return best_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_epochs", type=int, default=S1_EP)
    ap.add_argument("--s2_epochs", type=int, default=S2_EP)
    ap.add_argument("--s3_epochs", type=int, default=S3_EP)
    args = ap.parse_args()

    print(f"\nbuilding {ARCH}")
    model = build_model(len(classes)).to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {total:.1f}M")

    ema = ModelEMA(model, decay=EMA_DECAY)

    # stage 1 - 224px
    best = run_stage(model, ema, S1_IMG, args.s1_epochs,
                     S1_LR, S1_LRMIN, S1_WU, tag="S1")

    # load best before stage 2
    ckpt = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    ema = ModelEMA(model, decay=EMA_DECAY)

    # stage 2 - 320px
    best = run_stage(model, ema, S2_IMG, args.s2_epochs,
                     S2_LR, S2_LRMIN, S2_WU, best_acc=best, tag="S2")

    # load best before stage 3
    ckpt = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    ema = ModelEMA(model, decay=EMA_DECAY)

    # stage 3 - 448px + swa
    best = run_stage(model, ema, S3_IMG, args.s3_epochs,
                     S3_LR, S3_LRMIN, S3_WU, best_acc=best,
                     tag="S3", use_swa=True)

    # 20-view tta eval on best checkpoint
    print("\nfinal 20-view tta eval...")
    ckpt = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    val_ds_tta     = KhanaDataset(val_paths, val_labels, make_val_tf(S3_IMG))
    val_loader_tta = DataLoader(val_ds_tta, BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
    tta_acc1, tta_acc5 = evaluate(model, val_loader_tta, tta=True, img_size=S3_IMG)
    print(f"\nfinal tta val@1: {tta_acc1:.4f} | val@5: {tta_acc5:.4f}")
    print(f"best saved (no tta): {best:.4f}")
    print(f"tta gain: +{(tta_acc1-best)*100:.2f}%")
