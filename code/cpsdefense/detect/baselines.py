import numpy as np

from cpsdefense.detect.scoring import FeatureScaler, topk_mean


class RobustZScore:
    name = "zscore"

    def __init__(self, topk=(1, 3, 10), std_floor=0.1):
        self.topk = list(topk)
        self.std_floor = float(std_floor)

    def fit(self, train, val=None):
        self.scaler = FeatureScaler("robust", self.std_floor).fit(train["X"])
        columns = train.get("columns")
        self.floored_columns = [columns[i] if columns is not None else i for i in self.scaler.std_floored]
        return self

    def scores(self, X, segment=None):
        z = np.abs(self.scaler.transform(X))
        return {f"top{k}": topk_mean(z, k) for k in self.topk}

    def describe(self):
        return {"detector": self.name, "topk": self.topk, "std_floor": self.std_floor,
                "std_floored_columns": self.floored_columns, "scaler": self.scaler.describe()}


class PCAReconstruction:
    name = "pca"

    def __init__(self, variance=(0.9, 0.95, 0.99)):
        self.variance = [float(v) for v in variance]
        if any(not 0 < v <= 1 for v in self.variance):
            raise ValueError("explained-variance levels must be in (0, 1]")

    def fit(self, train, val=None):
        self.scaler = FeatureScaler("standard").fit(train["X"])
        Z = self.scaler.transform(train["X"]).astype(np.float64)
        self.mean = Z.mean(axis=0)
        Z -= self.mean
        evals, evecs = np.linalg.eigh(Z.T @ Z / (Z.shape[0] - 1))
        order = np.argsort(evals)[::-1]
        evals, self.components = np.clip(evals[order], 0, None), evecs[:, order]
        self.explained_ratio = evals / evals.sum()
        cum = np.cumsum(self.explained_ratio)
        f = cum.size
        self.n_components = {v: min(int(np.searchsorted(cum, v - 1e-12)) + 1, f) for v in self.variance}
        return self

    def scores(self, X, segment=None):
        Z = self.scaler.transform(X).astype(np.float64) - self.mean
        out = {}
        for v, k in self.n_components.items():
            P = self.components[:, :k]
            R = Z - (Z @ P) @ P.T
            out[f"var{v:g}"] = np.einsum("ij,ij->i", R, R)
        return out

    def describe(self):
        return {"detector": self.name, "variance": self.variance,
                "n_components": {f"{v:g}": k for v, k in self.n_components.items()},
                "explained_ratio_top10": self.explained_ratio[:10].tolist(),
                "scaler": self.scaler.describe()}
