"""Train a deep MNIST autoencoder with any pytorch-opt optimizer.

The 784-512-256-30-256-512-784 bottleneck autoencoder is the classic
second-order benchmark (Hinton & Salakhutdinov 2006; Martens 2010): its
curvature is pathological enough that ordinary first-order optimizers converge
slowly to visibly worse reconstruction loss, so optimizer differences actually
show up.

Install as a third party (no clone needed):

    pip install git+https://github.com/kavinravi/pytorch-opt.git

Run:

    python train_mnist_autoencoder.py --optimizer shampoo
    python train_mnist_autoencoder.py --optimizer adamw      # baseline
    python train_mnist_autoencoder.py --optimizer hessianfree --epochs 2

Drop-in compatibility (see the OPTIMIZERS table and train_step below):
  * muon / shampoo / soap / psgd     - literal AdamW drop-ins (backward+step)
  * kfac / ekfac                     - same loop, but constructed from the
                                       MODEL (they install module hooks)
  * trustncg / hessianfree /
    adahessian / sophia              - closure-based, like torch's LBFGS: the
                                       closure returns the loss WITH its graph
                                       (no backward() inside); GGN-based ones
                                       return (loss, outputs)
`train_step` handles all three protocols generically via the
`requires_closure` / `curvature` attributes.
"""

import argparse
import time

import torch
from torch import nn
from torchvision import datasets, transforms

from pytorch_opt import (AdaHessian, EKFAC, HessianFree, KFAC, Muon, PSGD,
                         SOAP, Shampoo, Sophia, TrustNCG)


def mse_half(out, target):
    """pytorch-opt's MSE convention: 0.5 * sum of squares / batch (docs/design.md).

    Using this convention matters for the GGN-based closure optimizers
    (curvature='ggn', ggn_loss='mse') so their curvature matches the loss.
    """
    return 0.5 * (out - target).pow(2).sum() / out.shape[0]


# Starting points, not tuned optima -- second-order lrs are NOT Adam lrs.
OPTIMIZERS = {
    # ---- literal AdamW drop-ins (params in, backward + step) ----
    "adamw":   lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3),
    "muon":    lambda m: Muon(m.parameters(), lr=0.02, adamw_lr=1e-3),
    "shampoo": lambda m: Shampoo(m.parameters(), lr=3e-3, graft="adagrad",
                                 momentum=0.9, precondition_frequency=20),
    "soap":    lambda m: SOAP(m.parameters(), lr=3e-3, precondition_frequency=10),
    "psgd":    lambda m: PSGD(m.parameters(), lr=1e-3, precond_lr=0.1,
                              grad_clip_max_norm=100.0),
    # ---- same training loop, but constructed from the model (hooks) ----
    "kfac":    lambda m: KFAC(m, lr=1e-3, damping=3e-2, inv_every=20,
                              max_grad_norm=10.0),
    "ekfac":   lambda m: EKFAC(m, lr=1e-3, damping=3e-2, inv_every=20,
                               max_grad_norm=10.0),
    # ---- closure-based (curvature products / extra evals per step) ----
    "adahessian":  lambda m: AdaHessian(m.parameters(), lr=3e-3, eps=1e-4,
                                        update_freq=5),
    "sophia":      lambda m: Sophia(m.parameters(), lr=1e-3, estimate_freq=10),
    "trustncg":    lambda m: TrustNCG(m.parameters(), delta0=1.0,
                                      curvature="ggn", max_cg_iter=30),
    "hessianfree": lambda m: HessianFree(m.parameters(), max_cg_iter=30),
    # NGD is omitted: it builds the dense Fisher (small models only; this
    # network has ~1M parameters).
}


def make_model():
    dims = [784, 512, 256, 30, 256, 512, 784]
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def train_step(model, opt, x):
    """One optimizer step; handles all three pytorch-opt protocols."""
    if getattr(opt, "requires_closure", False):
        def closure():
            out = model(x)
            loss = mse_half(out, x)
            if getattr(opt, "curvature", "hessian") == "ggn":
                return loss, out
            return loss
        loss = opt.step(closure)
    else:
        opt.zero_grad(set_to_none=True)
        loss = mse_half(model(x), x)
        loss.backward()
        opt.step()
    return float(loss)


@torch.no_grad()
def evaluate(model, loader, device):
    total, n = 0.0, 0
    for x, _ in loader:
        x = x.view(x.shape[0], -1).to(device)
        total += float(mse_half(model(x), x)) * x.shape[0]
        n += x.shape[0]
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="shampoo")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="limit batches per epoch (smoke tests)")
    ap.add_argument("--lr", type=float, default=None,
                    help="override the registry learning rate")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tfm = transforms.ToTensor()
    train = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                               shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1024)

    model = make_model().to(device)
    opt = OPTIMIZERS[args.optimizer](model)
    if args.lr is not None:
        for g in opt.param_groups:
            g["lr"] = args.lr
    print(f"{args.optimizer} on {device} | params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for i, (x, _) in enumerate(train_loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            x = x.view(x.shape[0], -1).to(device)
            running += train_step(model, opt, x)
            seen += 1
        test_loss = evaluate(model, test_loader, device)
        print(f"epoch {epoch}: train {running / seen:.4f} | "
              f"test {test_loss:.4f} | {time.perf_counter() - t0:.1f}s")

    diag = getattr(opt, "diagnostics", None)   # pytorch-opt optimizers only
    if diag:
        print("last-step diagnostics:", diag)


if __name__ == "__main__":
    main()
