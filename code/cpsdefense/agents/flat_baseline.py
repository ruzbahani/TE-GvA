from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

GROUP_KEYS = ("stale", "deviation", "stale_share", "max_score", "field_gap", "frozen")
LINK_KEYS = ("reports", "losses", "latest_loss", "latest_backup", "trace_healthy", "trace_backup")
MEAS_KEYS = ("stale", "deviation")


def _pos(vocab, source, label):
    return vocab["payload_layout"][source].index(label)


def _block(vocab, source, key):
    block = next(b for b in vocab["subject_blocks"][source] if b["key"] == key)
    return block["offset"], len(block["ids"])


def _latest(sel, time):
    return np.where(sel, time, -np.inf).argmax(1), sel.any(1)


def evidence_summary(batch, vocab):
    valid = np.asarray(batch.validity())
    source, payload = np.asarray(batch.source), np.asarray(batch.payload, np.float64)
    event_time, tau = np.asarray(batch.event_time), np.asarray(batch.decision_time)
    n, groups, links, names = valid.shape[0], vocab["groups"], vocab["links"], vocab["measurements"]
    rows = np.arange(n)
    src = {name: s for s, name in enumerate(vocab["sources"])}
    out = {"n_valid": valid.sum(1).astype(float)}

    sel = valid & (source == src["alert"])
    stale, dev = payload[..., _pos(vocab, "alert", "stale")] > 0.5, payload[..., _pos(vocab, "alert", "deviation")] > 0.5
    score = payload[..., _pos(vocab, "alert", "score")]
    off, k = _block(vocab, "alert", "subject")
    group = payload[..., off:off + k].argmax(-1)
    off, k = _block(vocab, "alert", "measurement")
    meas = payload[..., off:off + k].argmax(-1)
    g_stale, g_dev, g_share, g_score = (np.zeros((n, len(groups))) for _ in range(4))
    m_stale, m_dev = np.zeros((n, len(names))), np.zeros((n, len(names)))
    for j in range(len(names)):
        m_stale[:, j] = (sel & stale & (meas == j)).sum(1)
        m_dev[:, j] = (sel & dev & (meas == j)).sum(1)
    for g, name in enumerate(groups):
        in_g = sel & (group == g)
        g_stale[:, g], g_dev[:, g] = (in_g & stale).sum(1), (in_g & dev).sum(1)
        members = [names.index(m) for m in vocab["group_measurements"][name]]
        g_share[:, g] = (m_stale[:, members] > 0).mean(1)
        g_score[:, g] = np.where(in_g.any(1), np.where(in_g, score, -np.inf).max(1), np.nan)
    first = np.where(sel, event_time, np.inf).min(1)
    last = np.where(sel, event_time, -np.inf).max(1)
    out["alerts"] = sel.sum(1).astype(float)
    out["since_first_alert"] = np.where(sel.any(1), tau - first, np.nan)
    out["since_last_alert"] = np.where(sel.any(1), tau - last, np.nan)

    lk = {key: np.full((n, len(links)), np.nan) for key in LINK_KEYS}
    for name, keys in (("link_health", ("loss", "backup")), ("trace_link", ("healthy", "backup"))):
        s = valid & (source == src[name])
        off, k = _block(vocab, name, "subject")
        subject = payload[..., off:off + k].argmax(-1)
        a, b = (payload[..., _pos(vocab, name, key)] for key in keys)
        for li in range(len(links)):
            on = s & (subject == li)
            slot, has = _latest(on, event_time)
            if name == "link_health":
                lk["reports"][:, li] = on.sum(1)
                lk["losses"][:, li] = (on & (a > 0)).sum(1)
                lk["latest_loss"][:, li] = np.where(has, a[rows, slot], np.nan)
                lk["latest_backup"][:, li] = np.where(has, b[rows, slot], np.nan)
            else:
                lk["trace_healthy"][:, li] = np.where(has, a[rows, slot], np.nan)
                lk["trace_backup"][:, li] = np.where(has, b[rows, slot], np.nan)

    g_gap, g_frozen = np.full((n, len(groups)), np.nan), np.full((n, len(groups)), np.nan)
    for g, name in enumerate(groups):
        fc = f"field_check:{name}"
        layout = vocab["payload_layout"][fc]
        diff = [i for i, lab in enumerate(layout) if lab.endswith(":published_minus_field")]
        if diff:
            gap = np.abs(payload[..., diff]).max(-1)
        else:
            pub = [i for i, lab in enumerate(layout) if lab.endswith(":published")]
            fld = [i for i, lab in enumerate(layout) if lab.endswith(":field")]
            gap = np.abs(payload[..., pub] - payload[..., fld]).max(-1)
        slot, has = _latest(valid & (source == src[fc]), event_time)
        g_gap[:, g] = np.where(has, gap[rows, slot], np.nan)
        win = f"window:{name}"
        frozen = [i for i, lab in enumerate(vocab["payload_layout"][win]) if lab.endswith(":frac_unchanged")]
        slot, has = _latest(valid & (source == src[win]), event_time)
        g_frozen[:, g] = np.where(has, payload[rows, slot][:, frozen].mean(-1), np.nan)

    slot, has = _latest(valid & (source == src["operator_report"]), event_time)
    causes = len(vocab["payload_layout"]["operator_report"])
    out["operator"] = np.where(has[:, None], payload[rows, slot, :causes], 0.0)
    out["operator_any"] = has.astype(float)
    out["group"] = dict(zip(GROUP_KEYS, (g_stale, g_dev, g_share, g_score, g_gap, g_frozen)))
    out["link"] = lk
    out["measurement"] = {"stale": m_stale, "deviation": m_dev}
    if "receipt" in src:
        out["receipt"] = receipt_summary(valid & (source == src["receipt"]), payload, event_time, vocab)
    return out


def receipt_summary(sel, payload, event_time, vocab):
    tools, statuses = vocab["tools"], vocab["receipt_statuses"]
    n, rows = sel.shape[0], np.arange(sel.shape[0])
    tool = payload[..., [_pos(vocab, "receipt", f"tool={t}") for t in tools]].argmax(-1)
    status = payload[..., [_pos(vocab, "receipt", f"status={s}") for s in statuses]].argmax(-1)
    off, k = _block(vocab, "receipt", "subject")
    subject = payload[..., off:off + k].argmax(-1)
    ok, running = statuses.index("succeeded"), statuses.index("running")
    count, latest_ok = np.zeros((n, len(tools))), np.full((n, len(tools)), np.nan)
    for t in range(len(tools)):
        on = sel & (tool == t)
        slot, has = _latest(on, event_time)
        count[:, t] = on.sum(1)
        latest_ok[:, t] = np.where(has, (status[rows, slot] == ok).astype(float), np.nan)
    at_n = np.stack([(sel & (subject == j)).sum(1) for j in range(k)], 1).astype(float)
    at_running = np.stack([(sel & (subject == j) & (status == running)).any(1) for j in range(k)], 1).astype(float)
    at_status = np.full((n, len(tools), k), np.nan)
    for t, j in {(int(a), int(b)) for a, b in zip(tool[sel], subject[sel])}:
        slot, has = _latest(sel & (tool == t) & (subject == j), event_time)
        at_status[:, t, j] = np.where(has, status[rows, slot], np.nan)
    return {"count": count, "latest_ok": latest_ok, "at_n": at_n, "at_running": at_running, "at_status": at_status}


def decision_matrix(summary, vocab):
    cols, names = [], []
    for key in ("n_valid", "alerts", "since_first_alert", "since_last_alert", "operator_any"):
        cols.append(summary[key][:, None])
        names.append(key)
    cols.append(summary["operator"])
    names += [f"operator:{c}" for c in vocab["payload_layout"]["operator_report"]]
    for key in GROUP_KEYS:
        cols.append(summary["group"][key])
        names += [f"{key}:{g}" for g in vocab["groups"]]
    for key in LINK_KEYS:
        cols.append(summary["link"][key])
        names += [f"{key}:{link}" for link in vocab["links"]]
    if "receipt" in summary:
        for key in ("count", "latest_ok"):
            cols.append(summary["receipt"][key])
            names += [f"receipt_{key}:{t}" for t in vocab["tools"]]
    return np.concatenate(cols, 1), names


def candidate_targets(cand_features, vocab):
    lo, hi = vocab["action_blocks"]["target"]
    targets = [t for k in vocab["target_kinds"] for t in vocab["targets"][k]]
    tid = np.asarray(cand_features)[:, lo:hi].argmax(1)
    groups, links, names = vocab["groups"], vocab["links"], vocab["measurements"]
    group_of = {m: g for g, ms in vocab["group_measurements"].items() for m in ms}
    table = []
    for t in targets:
        g = link = m = -1
        if t in links:
            g, link = groups.index(t.split(":", 1)[1]), links.index(t)
        elif t.startswith("source:"):
            g = groups.index(t[7:])
            link = links.index(f"sensor:{t[7:]}")
        elif t in names:
            g, m = groups.index(group_of[t]), names.index(t)
            link = links.index(f"sensor:{group_of[t]}")
        table.append((g, link, m))
    return np.asarray(table, np.int64)[tid]


def feature_matrix(summary, cand_features, cand_decision, vocab):
    cand_features = np.asarray(cand_features, np.float64)
    dec = np.asarray(cand_decision, np.int64)
    X_dec, dec_names = decision_matrix(summary, vocab)
    tg = candidate_targets(cand_features, vocab)
    aligned, names = [], []
    for kind, idx, keys in (("group", tg[:, 0], GROUP_KEYS), ("link", tg[:, 1], LINK_KEYS),
                            ("measurement", tg[:, 2], MEAS_KEYS)):
        for key in keys:
            table = summary[kind][key]
            aligned.append(np.where(idx >= 0, table[dec, np.maximum(idx, 0)], np.nan))
            names.append(f"target_{kind}:{key}")
    if "receipt" in summary:
        r = summary["receipt"]
        lo, hi = vocab["action_blocks"]["target"]
        subjects = next(b for b in vocab["subject_blocks"]["receipt"] if b["key"] == "subject")["ids"]
        targets = [t for k in vocab["target_kinds"] for t in vocab["targets"][k]]
        subj = np.asarray([subjects.index(t) if t in subjects else -1 for t in targets])[
            cand_features[:, lo:hi].argmax(1)]
        lo, hi = vocab["action_blocks"]["tool"]
        tool = cand_features[:, lo:hi].argmax(1)
        s = np.maximum(subj, 0)
        for name, col in (("n", r["at_n"][dec, s]), ("running", r["at_running"][dec, s]),
                          ("same_tool_status", r["at_status"][dec, tool, s])):
            aligned.append(np.where(subj >= 0, col, np.nan))
            names.append(f"target_receipt:{name}")
    X = np.concatenate([cand_features, X_dec[dec], np.stack(aligned, 1)], 1)
    return X, list(vocab["action_columns"]) + dec_names + names


def local_argmin(pred, cand_decision, n_decisions):
    dec = np.asarray(cand_decision, np.int64)
    order = np.argsort(dec, kind="stable")
    counts = np.bincount(dec, minlength=n_decisions)
    slot = np.empty(dec.size, np.int64)
    slot[order] = np.arange(dec.size) - np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    best = np.full(n_decisions, np.inf)
    np.minimum.at(best, dec, pred)
    out = np.full(n_decisions, -1, np.int64)
    hit = pred == best[dec]
    for i in np.flatnonzero(hit)[::-1]:
        out[dec[i]] = slot[i]
    return out


@dataclass
class FlatBaseline:
    learning_rate: float = 0.1
    max_leaf_nodes: int = 31
    max_iter: int = 300
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    center: bool = True
    seed: int = 0
    reg: object = field(default=None, repr=False)
    clf: object = field(default=None, repr=False)
    trip_constant: float = 0.0

    def _params(self):
        return {"learning_rate": self.learning_rate, "max_leaf_nodes": self.max_leaf_nodes, "max_iter": self.max_iter,
                "min_samples_leaf": self.min_samples_leaf, "l2_regularization": self.l2_regularization,
                "early_stopping": False, "random_state": self.seed}

    def fit(self, X, J, p_trip, cand_decision, trip=True):
        J, p = np.asarray(J, np.float64), np.asarray(p_trip, np.float64)
        dec = np.asarray(cand_decision, np.int64)
        y = J
        if self.center:
            mean = np.bincount(dec, J) / np.maximum(np.bincount(dec), 1)
            y = J - mean[dec]
        self.reg = HistGradientBoostingRegressor(**self._params()).fit(X, y)
        self.trip_constant, self.clf = float(p.mean()), None
        if trip and 0 < p.mean() < 1:
            label = np.r_[np.ones(len(p)), np.zeros(len(p))]
            weight = np.r_[p, 1 - p]
            keep = weight > 0
            self.clf = HistGradientBoostingClassifier(**self._params()).fit(
                np.concatenate([X, X])[keep], label[keep], sample_weight=weight[keep])
        return self

    def predict(self, X):
        trip = self.clf.predict_proba(X)[:, 1] if self.clf is not None else np.full(len(X), self.trip_constant)
        return self.reg.predict(X), trip
