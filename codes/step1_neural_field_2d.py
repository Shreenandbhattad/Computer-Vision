import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path("E:/Computer Vision")
OUT_DIR     = PROJECT_DIR / "nerf_output" / "2d"
ITERS       = 3000
BATCH_SIZE  = 4096
LR          = 5e-4
L_PE        = 10
HIDDEN      = 256
N_LAYERS    = 4
LOG_EVERY   = 500
IMG_RESIZE  = 400


class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L
        self.register_buffer("freqs", 2.0 ** torch.arange(L))

    def forward(self, x):
        out = [x]
        for f in self.freqs:
            out.append(torch.sin(f * torch.pi * x))
            out.append(torch.cos(f * torch.pi * x))
        return torch.cat(out, dim=-1)


class NeuralField2D(nn.Module):
    def __init__(self, L=10, hidden=256, n_layers=4):
        super().__init__()
        self.pe  = PositionalEncoding(L)
        in_dim   = 2 + 4 * L
        layers   = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers  += [nn.Linear(hidden, 3), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, uv):
        return self.mlp(self.pe(uv))


class PixelDataset(Dataset):
    def __init__(self, img_path, resize=None):
        img = Image.open(img_path).convert("RGB")
        if resize:
            img = img.resize((resize, int(img.height * resize / img.width)), Image.BICUBIC)
        self.H, self.W = img.height, img.width
        img_np = np.array(img, dtype=np.float32) / 255.0
        ys, xs = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        coords = np.stack([xs / self.W, ys / self.H], axis=-1)
        self.coords = torch.from_numpy(coords.reshape(-1, 2).astype(np.float32))
        self.colors = torch.from_numpy(img_np.reshape(-1, 3))

    def __len__(self):        return len(self.coords)
    def __getitem__(self, i): return self.coords[i], self.colors[i]


def psnr(mse): return -10.0 * np.log10(mse + 1e-10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(PROJECT_DIR / "nerf_data" / "undistorted" / "IMG_0901.jpg"))
    ap.add_argument("--iters", type=int,   default=ITERS)
    ap.add_argument("--lr",    type=float, default=LR)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds     = PixelDataset(args.image, resize=IMG_RESIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)
    print(f"image: {ds.W}x{ds.H} | {len(ds)} pixels")

    model   = NeuralField2D(L=L_PE, hidden=HIDDEN, n_layers=N_LAYERS).to(device)
    optim   = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    psnr_log    = []
    loader_iter = iter(loader)
    model.train()

    for step in tqdm(range(1, args.iters + 1)):
        try:
            coords, colors = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            coords, colors = next(loader_iter)

        coords = coords.to(device)   # already float32
        colors = colors.to(device)
        pred   = model(coords)
        loss   = loss_fn(pred, colors)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % LOG_EVERY == 0 or step == 1:
            p = psnr(float(loss.item()))
            psnr_log.append((step, p))
            print(f"  step {step:5d} | mse {loss.item():.6f} | psnr {p:.2f} dB")

    # render full image
    print("\nrendering")
    model.eval()
    with torch.no_grad():
        chunks = [model(ds.coords[i:i+8192].to(device)).cpu()
                  for i in range(0, len(ds.coords), 8192)]
    pred_np  = torch.cat(chunks).numpy()
    rendered = (pred_np.reshape(ds.H, ds.W, 3) * 255).astype(np.uint8)
    gt       = (ds.colors.numpy().reshape(ds.H, ds.W, 3) * 255).astype(np.uint8)
    final_psnr = psnr(float(loss_fn(torch.from_numpy(pred_np), ds.colors).item()))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt);       axes[0].set_title("Ground Truth");  axes[0].axis("off")
    axes[1].imshow(rendered); axes[1].set_title(f"Neural Field (PSNR={final_psnr:.2f} dB)"); axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison.png", dpi=150)
    print(f"saved {OUT_DIR / 'comparison.png'}")

    steps_log, psnrs = zip(*psnr_log)
    plt.figure(figsize=(7, 4))
    plt.plot(steps_log, psnrs, marker="o")
    plt.xlabel("Iteration"); plt.ylabel("PSNR (dB)")
    plt.title("2D Neural Field Training"); plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "psnr_curve.png", dpi=120)
    print(f"saved {OUT_DIR / 'psnr_curve.png'}")
    print(f"\nFinal PSNR: {final_psnr:.2f} dB")
