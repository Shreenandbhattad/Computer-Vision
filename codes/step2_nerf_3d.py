import argparse, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v3 as iio

PROJECT_DIR = Path("E:/Computer Vision")

# config
OUT_DIR    = PROJECT_DIR / "nerf_output" / "3d"
N_ITERS    = 30_000
LR         = 5e-4
BATCH_RAYS = 1024    
N_COARSE   = 64      
N_FINE     = 128    
L_POS      = 10     
L_DIR      = 4
HIDDEN     = 256
N_LAYERS   = 8       
LOG_EVERY  = 1000
NEAR       = 0.02    # near plane in metres
FAR        = 1.0     # far plane in metres
CHUNK      = 1024 * 32


# positional encoding
class PE(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.register_buffer("freqs", 2.0 ** torch.arange(L))
    def out_dim(self, in_dim): return in_dim * (1 + 2 * self.L)
    def forward(self, x):
        out = [x]
        for f in self.freqs:
            out += [torch.sin(f * math.pi * x), torch.cos(f * math.pi * x)]
        return torch.cat(out, dim=-1)


# nerf mlp - skip at layer 4, view-dependent color head
class NeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, hidden=256, n_layers=8):
        super().__init__()
        self.pe_pos = PE(L_pos)
        self.pe_dir = PE(L_dir)
        pos_dim = self.pe_pos.out_dim(3)
        dir_dim = self.pe_dir.out_dim(3)
        self.skip = 4   # re-inject input at this layer

        layers = []
        in_d = pos_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_d, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_d = hidden + pos_dim if i == self.skip - 1 else hidden
        self.density_net = nn.ModuleList(layers)

        self.sigma_head = nn.Linear(hidden, 1)
        self.feat_head  = nn.Linear(hidden, hidden)

        # color depends on view direction
        self.rgb_net = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, pos, dirs):
        x   = self.pe_pos(pos)
        inp = x
        h   = x
        idx = 0
        for i, layer in enumerate(self.density_net):
            if isinstance(layer, nn.Linear):
                if idx == self.skip:
                    h = torch.cat([h, inp], dim=-1)
                h = layer(h)
                idx += 1
            else:
                h = layer(h)

        sigma = torch.relu(self.sigma_head(h)).squeeze(-1)
        feat  = self.feat_head(h)
        rgb   = self.rgb_net(torch.cat([feat, self.pe_dir(dirs)], dim=-1))
        return rgb, sigma


def get_rays(H, W, focal, c2w):
    device = c2w.device
    i, j   = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy"
    )

    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i),
    ], dim=-1)

    R      = c2w[:3, :3]
    rays_d = (dirs[..., None, :] * R).sum(-1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def sample_coarse(rays_o, rays_d, near, far, N_samples):
    # stratified sampling - one sample per bin with random offset
    N  = rays_o.shape[0]
    t  = torch.linspace(near, far, N_samples, device=rays_o.device)
    dt = (far - near) / N_samples
    t  = t.unsqueeze(0).expand(N, -1) + torch.rand(N, N_samples, device=rays_o.device) * dt
    pts = rays_o[:, None, :] + rays_d[:, None, :] * t[:, :, None]
    return pts, t


def sample_fine(rays_o, rays_d, t_coarse, weights, N_fine):
    # importance sampling - more samples where density is high
    weights  = weights + 1e-5
    pdf      = weights / weights.sum(-1, keepdim=True)
    cdf      = torch.cumsum(pdf, dim=-1)
    cdf      = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    u        = torch.rand(*cdf.shape[:-1], N_fine, device=rays_o.device)
    inds     = torch.searchsorted(cdf.detach(), u, right=True)
    below    = (inds - 1).clamp(min=0)
    above    = inds.clamp(max=cdf.shape[-1] - 1)
    inds_g   = torch.stack([below, above], dim=-1)

    cdf_g    = torch.gather(cdf,      1, inds_g.reshape(*inds_g.shape[:-2], -1)).reshape(*inds_g.shape)
    bins_g   = torch.gather(t_coarse, 1, inds_g.reshape(*inds_g.shape[:-2], -1)).reshape(*inds_g.shape)

    denom    = cdf_g[..., 1] - cdf_g[..., 0]
    denom    = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t_fine   = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])

    t_all, _ = torch.sort(torch.cat([t_coarse, t_fine], dim=-1), dim=-1)
    pts      = rays_o[:, None, :] + rays_d[:, None, :] * t_all[:, :, None]
    return pts, t_all


def volume_render(rgb_pts, sigma_pts, t_vals, rays_d):
    # classic nerf rendering: integrate color along ray
    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    last   = torch.full((deltas.shape[0], 1), 1e10, device=deltas.device)
    deltas = torch.cat([deltas, last], dim=-1)

    alpha  = 1.0 - torch.exp(-sigma_pts * deltas)

    T = torch.cumprod(
        torch.cat([torch.ones(*alpha.shape[:1], 1, device=alpha.device),
                   1.0 - alpha[:, :-1] + 1e-10], dim=-1), dim=-1
    )
    weights = T * alpha
    rgb     = (weights.unsqueeze(-1) * rgb_pts).sum(dim=1)
    return rgb, weights


def render_rays(nerf_coarse, nerf_fine, rays_o, rays_d,
                near, far, N_coarse, N_fine, chunk=CHUNK):
    N = rays_o.shape[0]
    rgb_fine_all   = []
    rgb_coarse_all = []

    for i in range(0, N, chunk):
        ro = rays_o[i:i+chunk]
        rd = rays_d[i:i+chunk]
        n  = ro.shape[0]

        # coarse pass
        pts_c, t_c   = sample_coarse(ro, rd, near, far, N_coarse)
        dirs_c       = rd[:, None, :].expand_as(pts_c)
        rgb_c, sig_c = nerf_coarse(pts_c.reshape(-1, 3), dirs_c.reshape(-1, 3))
        rgb_c  = rgb_c.reshape(n, N_coarse, 3)
        sig_c  = sig_c.reshape(n, N_coarse)
        rgb_c_rendered, w_c = volume_render(rgb_c, sig_c, t_c, rd)

        # fine pass using coarse weights for importance sampling
        with torch.no_grad():
            w_mid = (w_c[:, :-1] + w_c[:, 1:]) / 2
        pts_f, t_f   = sample_fine(ro, rd, t_c, w_mid, N_fine)
        dirs_f       = rd[:, None, :].expand_as(pts_f)
        rgb_f, sig_f = nerf_fine(pts_f.reshape(-1, 3), dirs_f.reshape(-1, 3))
        rgb_f  = rgb_f.reshape(n, -1, 3)
        sig_f  = sig_f.reshape(n, -1)
        rgb_f_rendered, _ = volume_render(rgb_f, sig_f, t_f, rd)

        rgb_coarse_all.append(rgb_c_rendered)
        rgb_fine_all.append(rgb_f_rendered)

    return torch.cat(rgb_fine_all, 0), torch.cat(rgb_coarse_all, 0)


def psnr(mse): return -10.0 * np.log10(mse + 1e-10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default=str(PROJECT_DIR / "nerf_data" / "thali_dataset.npz"))
    ap.add_argument("--iters", type=int,   default=N_ITERS)
    ap.add_argument("--lr",    type=float, default=LR)
    ap.add_argument("--near",  type=float, default=NEAR)
    ap.add_argument("--far",   type=float, default=FAR)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load dataset
    data   = np.load(args.data)
    imgs   = torch.from_numpy(data["images_train"].astype(np.float32) / 255.0)
    c2ws   = torch.from_numpy(data["c2ws_train"].astype(np.float32))
    focal  = float(data["focal"])
    N, H, W, _ = imgs.shape
    print(f"train: {N} images | {W}x{H} | focal={focal:.1f}px")

    imgs_val = torch.from_numpy(data["images_val"].astype(np.float32) / 255.0)
    c2ws_val = torch.from_numpy(data["c2ws_val"].astype(np.float32))

    # precompute all training rays
    print("computing rays...")
    all_rays_o = []
    all_rays_d = []
    all_rgb    = []
    for i in range(N):
        ro, rd = get_rays(H, W, focal, c2ws[i])
        all_rays_o.append(ro)
        all_rays_d.append(rd)
        all_rgb.append(imgs[i].reshape(-1, 3))
    all_rays_o = torch.cat(all_rays_o, 0)
    all_rays_d = torch.cat(all_rays_d, 0)
    all_rgb    = torch.cat(all_rgb,    0)
    print(f"total rays: {len(all_rays_o):,}")

    # models + optimizer
    nerf_c = NeRF(L_pos=L_POS, L_dir=L_DIR, hidden=HIDDEN, n_layers=N_LAYERS).to(device)
    nerf_f = NeRF(L_pos=L_POS, L_dir=L_DIR, hidden=HIDDEN, n_layers=N_LAYERS).to(device)
    params  = list(nerf_c.parameters()) + list(nerf_f.parameters())
    optim   = torch.optim.Adam(params, lr=args.lr)

    # cosine lr decay
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr * 0.01)

    loss_fn  = nn.MSELoss()
    psnr_log = []
    t0 = time.time()

    for step in tqdm(range(1, args.iters + 1)):
        idx    = torch.randint(0, len(all_rays_o), (BATCH_RAYS,))
        ro     = all_rays_o[idx].to(device)
        rd     = all_rays_d[idx].to(device)
        target = all_rgb[idx].to(device)

        rgb_f, rgb_c = render_rays(nerf_c, nerf_f, ro, rd,
                                   args.near, args.far, N_COARSE, N_FINE)

        loss = loss_fn(rgb_f, target) + 0.5 * loss_fn(rgb_c, target)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        sched.step()

        if step % LOG_EVERY == 0 or step == 1:
            p = psnr(float(loss_fn(rgb_f.detach(), target).item()))
            psnr_log.append((step, p))
            elapsed = time.time() - t0
            print(f"  step {step:6d} | loss {loss.item():.5f} | psnr {p:.2f} dB | {elapsed:.0f}s")

    # render val image
    print("\nrendering val image")
    nerf_c.eval(); nerf_f.eval()
    with torch.no_grad():
        ro, rd = get_rays(H, W, focal, c2ws_val[0].to(device))
        rgb_out, _ = render_rays(nerf_c, nerf_f, ro, rd,
                                 args.near, args.far, N_COARSE, N_FINE)
    rendered = (rgb_out.cpu().numpy().reshape(H, W, 3) * 255).clip(0,255).astype(np.uint8)
    gt       = (imgs_val[0].numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt);       axes[0].set_title("Ground Truth"); axes[0].axis("off")
    axes[1].imshow(rendered); axes[1].set_title("NeRF Render");  axes[1].axis("off")
    fig.tight_layout(); fig.savefig(OUT_DIR / "val_render.png", dpi=150)
    iio.imwrite(OUT_DIR / "val_render.png", rendered)
    print(f"saved {OUT_DIR / 'val_render.png'}")

    print("rendering spiral video")
    frames = []
    c2w_0  = c2ws_val[0].numpy()
    for fi in tqdm(range(60)):
        angle = fi / 60 * 2 * math.pi
        rot   = np.array([[math.cos(angle), 0, math.sin(angle), 0],
                           [0,              1, 0,               0],
                           [-math.sin(angle),0,math.cos(angle), 0],
                           [0,              0, 0,               1]], dtype=np.float32)
        c2w_i = torch.from_numpy(rot @ c2w_0).to(device)
        with torch.no_grad():
            ro, rd = get_rays(H, W, focal, c2w_i)
            rgb_i, _ = render_rays(nerf_c, nerf_f, ro, rd,
                                   args.near, args.far, N_COARSE, N_FINE)
        frames.append((rgb_i.cpu().numpy().reshape(H, W, 3) * 255).clip(0,255).astype(np.uint8))

    iio.imwrite(OUT_DIR / "spiral.gif", frames, duration=1/24, loop=0)
    print(f"saved {OUT_DIR / 'spiral.gif'}")

    steps_log, psnrs = zip(*psnr_log)
    plt.figure(figsize=(8, 4))
    plt.plot(steps_log, psnrs, marker="o")
    plt.xlabel("Iteration"); plt.ylabel("PSNR (dB)"); plt.title("NeRF Training PSNR"); plt.grid(True)
    plt.tight_layout(); plt.savefig(OUT_DIR / "psnr_curve.png", dpi=120)
    print(f"saved -> {OUT_DIR / 'psnr_curve.png'}")

    torch.save({"nerf_c": nerf_c.state_dict(), "nerf_f": nerf_f.state_dict()},
               OUT_DIR / "nerf_weights.pt")
    print(f"saved weights {OUT_DIR / 'nerf_weights.pt'}")
