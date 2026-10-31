import copy
import time

import numpy as np
import torch
from torch import nn

from cpsdefense.data.splits import window_starts
from cpsdefense.detect.scoring import FeatureScaler, topk_mean
from cpsdefense.utils.seeding import spawn_generators


class GRUNet(nn.Module):
    def __init__(self, n_features, hidden=128, layers=2, dropout=0.1, clip=10.0):
        super().__init__()
        self.clip = float(clip)
        sizes = [n_features] + [hidden] * layers
        self.grus = nn.ModuleList(nn.GRU(a, b, batch_first=True) for a, b in zip(sizes[:-1], sizes[1:]))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x):
        h = x.clamp(-self.clip, self.clip)
        for i, gru in enumerate(self.grus):
            h, _ = gru(self.drop(h) if i else h)
        return x[:, -1] + self.head(h[:, -1])


class GRUForecaster:
    name = "gru"

    def __init__(self, window=60, hidden=128, layers=2, dropout=0.1, lr=1e-3, weight_decay=1e-4,
                 batch_size=512, steps_per_epoch=500, max_epochs=40, patience=5, lr_patience=2, lr_factor=0.5,
                 max_minutes=9.0, clip=10.0, residual_floor=1e-4, topk=(1, 3, 10, "all"), eval_batch=8192,
                 seed=0, device="cuda", log=None):
        if window < 2:
            raise ValueError("window must be >= 2 (context + target)")
        self.window = int(window)
        self.hidden, self.layers, self.dropout, self.clip = int(hidden), int(layers), float(dropout), float(clip)
        self.lr, self.weight_decay = float(lr), float(weight_decay)
        self.batch_size, self.steps_per_epoch = int(batch_size), int(steps_per_epoch)
        self.max_epochs, self.patience, self.max_minutes = int(max_epochs), int(patience), float(max_minutes)
        self.lr_patience, self.lr_factor = int(lr_patience), float(lr_factor)
        self.residual_floor, self.topk, self.eval_batch = float(residual_floor), list(topk), int(eval_batch)
        self.seed = int(seed)
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.log = log or (lambda *_: None)

    def _windows(self, X, starts):
        idx = starts[:, None] + torch.arange(self.window, device=X.device)
        w = X[idx]
        return w[:, :-1], w[:, -1]

    def _prepare(self, split):
        X = torch.as_tensor(self.scaler.transform(split["X"]), device=self.device)
        starts = window_starts(split["segment"], self.window)
        return X, torch.as_tensor(starts, device=self.device)

    @torch.no_grad()
    def _predict(self, X, starts):
        self.model.eval()
        out = []
        for i in range(0, starts.numel(), self.eval_batch):
            x, y = self._windows(X, starts[i:i + self.eval_batch])
            out.append(y - self.model(x))
        return torch.cat(out) if out else torch.zeros(0, X.shape[1], device=X.device)

    def fit(self, train, val):
        self.scaler = FeatureScaler("standard").fit(train["X"])
        Xtr, starts_tr = self._prepare(train)
        Xva, starts_va = self._prepare(val)
        if starts_tr.numel() == 0 or starts_va.numel() == 0:
            raise ValueError("no complete windows in train or val")
        rng = spawn_generators(self.seed, ["detector"])["detector"]
        torch.manual_seed(self.seed)
        self.model = GRUNet(Xtr.shape[1], self.hidden, self.layers, self.dropout, self.clip).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=self.lr_factor, patience=self.lr_patience)
        best, best_state, wait, history, stop = np.inf, None, 0, [], "max_epochs"
        t0 = time.time()
        for epoch in range(self.max_epochs):
            self.model.train()
            pick = torch.as_tensor(rng.integers(0, starts_tr.numel(), size=(self.steps_per_epoch, self.batch_size)),
                                   device=self.device)
            total = torch.zeros((), device=self.device)
            for step in range(self.steps_per_epoch):
                x, y = self._windows(Xtr, starts_tr[pick[step]])
                loss = torch.mean((self.model(x) - y) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total += loss.detach()
            val_loss = float(torch.mean(self._predict(Xva, starts_va) ** 2))
            train_loss = float(total) / self.steps_per_epoch
            elapsed = time.time() - t0
            lr = opt.param_groups[0]["lr"]
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr,
                            "elapsed_s": elapsed})
            self.log(f"  [gru seed {self.seed}] epoch {epoch:2d} train {train_loss:.5f} val {val_loss:.5f} "
                     f"lr {lr:.1e} ({elapsed:.0f} s)")
            plateau.step(val_loss)
            if val_loss < best:
                best, best_state, wait = val_loss, copy.deepcopy(self.model.state_dict()), 0
            else:
                wait += 1
            if wait >= self.patience:
                stop = "patience"
                break
            if elapsed > self.max_minutes * 60:
                stop = "time_limit"
                break
        self.model.load_state_dict(best_state)
        ms = self._predict(Xva, starts_va).pow(2).mean(dim=0).double().cpu().numpy()
        self.val_mse = ms
        self.residual_scale = np.maximum(ms, self.residual_floor)
        self.train_info = {"history": history, "best_val_loss": best, "stop_reason": stop,
                           "epochs": len(history), "train_seconds": time.time() - t0,
                           "n_train_windows": int(starts_tr.numel()), "n_val_windows": int(starts_va.numel()),
                           "floored_features": np.flatnonzero(ms < self.residual_floor).tolist()}
        return self

    def residuals(self, X, segment):
        Xs, starts = self._prepare({"X": X, "segment": segment})
        out = np.full((len(X), Xs.shape[1]), np.nan, np.float32)
        if starts.numel():
            out[starts.cpu().numpy() + self.window - 1] = self._predict(Xs, starts).cpu().numpy()
        return out

    def scores(self, X, segment):
        r = self.residuals(X, segment) ** 2 / self.residual_scale.astype(np.float32)
        return {f"top{k}": topk_mean(r, k) for k in self.topk}

    def describe(self):
        return {"detector": self.name, "window": self.window, "hidden": self.hidden, "layers": self.layers,
                "dropout": self.dropout, "clip": self.clip, "lr": self.lr, "weight_decay": self.weight_decay,
                "batch_size": self.batch_size, "steps_per_epoch": self.steps_per_epoch,
                "lr_patience": self.lr_patience, "lr_factor": self.lr_factor,
                "residual_floor": self.residual_floor, "topk": self.topk, "seed": self.seed,
                "device": str(self.device), "n_parameters": sum(p.numel() for p in self.model.parameters()),
                "scaler": self.scaler.describe(), "residual_scale": self.residual_scale.tolist(),
                "train": self.train_info}
