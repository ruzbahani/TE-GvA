import numpy as np

from cpsdefense.agents.flat_baseline import evidence_summary, feature_matrix


class SummaryDescriptor:
    def __init__(self, vocab, clip=10.0):
        self.vocab = vocab
        self.clip = float(clip)
        self.n_action = len(vocab["action_columns"])
        self.names = None
        self.mean = self.std = self.miss_cols = self.support = None

    def raw(self, batch, cand_features, cand_decision):
        batch = batch.numpy() if batch.is_torch else batch
        valid_only = batch.replace(present=batch.validity()).pad_absent()
        cand_features = np.asarray(cand_features)
        if cand_features.shape[1] != self.n_action:
            raise ValueError(f"cand_features have {cand_features.shape[1]} columns, the vocabulary {self.n_action}")
        X, names = feature_matrix(evidence_summary(valid_only, self.vocab), cand_features, cand_decision, self.vocab)
        return X[:, self.n_action:], names[self.n_action:]

    def fit(self, S, names):
        S = np.asarray(S, np.float64)
        with np.errstate(invalid="ignore"), _quiet():
            mean, std = np.nanmean(S, 0), np.nanstd(S, 0)
        self.mean = np.nan_to_num(mean).astype(np.float64)
        std = np.nan_to_num(std)
        self.std = np.where(std > 1e-6, std, 1.0).astype(np.float64)
        self.miss_cols = np.isnan(S).any(0)
        self.support = ~np.isnan(S).all(0)
        self.names = list(names)
        return self

    @property
    def fitted(self):
        return self.mean is not None

    @property
    def width(self):
        return len(self.names) + int(self.miss_cols.sum())

    def column_names(self):
        return [f"summary:{n}" for n in self.names] + [f"missing:{n}" for n, m in zip(self.names, self.miss_cols) if m]

    def transform(self, S, names=None):
        if not self.fitted:
            raise ValueError("SummaryDescriptor is not fitted")
        if names is not None and list(names) != self.names:
            raise ValueError("summary columns differ from the fitted ones")
        S = np.where(self.support, np.asarray(S, np.float64), np.nan)
        Z = np.clip(np.nan_to_num((S - self.mean) / self.std), -self.clip, self.clip)
        return np.concatenate([Z, np.isnan(S[:, self.miss_cols]).astype(np.float64)], 1).astype(np.float32)

    def features(self, batch, cand_features, cand_decision):
        S, names = self.raw(batch, cand_features, cand_decision)
        return np.concatenate([np.asarray(cand_features, np.float32), self.transform(S, names)], 1)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "miss_cols": self.miss_cols, "support": self.support,
                "names": np.asarray(self.names), "clip": np.asarray(self.clip), "n_action": np.asarray(self.n_action)}

    def load_state_dict(self, state):
        if int(state["n_action"]) != self.n_action:
            raise ValueError("summary descriptor was fitted for another action vocabulary")
        self.mean, self.std = np.asarray(state["mean"], np.float64), np.asarray(state["std"], np.float64)
        self.miss_cols = np.asarray(state["miss_cols"], bool)
        self.support = np.asarray(state["support"], bool) if "support" in state else np.ones_like(self.miss_cols)
        self.names = [str(n) for n in state["names"]]
        self.clip = float(state["clip"])
        return self

    def save(self, path):
        np.savez(path, **self.state_dict())

    @classmethod
    def load(cls, path, vocab):
        with np.load(path, allow_pickle=False) as z:
            return cls(vocab).load_state_dict({k: z[k] for k in z.files})


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)
