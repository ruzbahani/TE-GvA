import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpsdefense.experience.dataset import KINDS
from cpsdefense.paths import ROOT

OUT = ROOT / "paper" / "figures"
SETS = {"natural": "data/processed/tep_decisions/v2-natural", "stress": "data/processed/tep_decisions/v2-stress"}
C = {"hindsight": "#b8b8b8", "belief": "#1f5fa8", "natural": "#1f5fa8", "stress": "#c0392b", "gbm": "#2e7d32",
     "net": "#6a3d9a", "base": "#8c6d31", "ref": "#555555", "light": "#d9e4f2"}
FULL, HALF = 6.5, 3.15


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.dpi": 300, "axes.titleweight": "normal", "axes.titlelocation": "left"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def panel(ax, letter, title=""):
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=8.5, pad=4)


def load_dev(root):
    keys_dec = ("dec_family", "dec_posterior", "dec_kind", "dec_backup_available", "dec_episode_seed")
    keys_cand = ("y_J", "y_J_belief", "y_observe", "cand_decision")
    out = {k: [] for k in (*keys_dec, *keys_cand)}
    offset = 0
    for split in ("train", "val"):
        with np.load(ROOT / root / f"{split}.npz", allow_pickle=False) as a:
            n = len(a["dec_tau"])
            for k in keys_dec:
                out[k].append(a[k])
            for k in keys_cand:
                out[k].append(a[k] + offset if k == "cand_decision" else a[k])
        offset += n
    return {k: np.concatenate(v) for k, v in out.items()}


def family_cause(d):
    link, module = KINDS.index("link_fault"), KINDS.index("io_module_fault")
    fam = d["dec_family"].astype(bool)
    ep_kind = {}
    for ep, k in zip(d["dec_episode_seed"][fam], d["dec_kind"][fam]):
        if k in (link, module):
            ep_kind[int(ep)] = int(k)
    out = np.zeros(len(fam), np.int64)
    for i in np.flatnonzero(fam):
        out[i] = 1 if ep_kind[int(d["dec_episode_seed"][i])] == link else -1
    return out


def per_decision(d, key):
    n = len(d["dec_family"])
    dec, J, obs = d["cand_decision"], d[key], d["y_observe"].astype(bool)
    omin, amin = np.full(n, np.inf), np.full(n, np.inf)
    np.minimum.at(omin, dec[obs], J[obs])
    np.minimum.at(amin, dec[~obs], J[~obs])
    return omin, amin


def belief_regret_of_hindsight_choice(d):
    n = len(d["dec_family"])
    dec = d["cand_decision"]
    best_h = np.full(n, np.inf)
    np.minimum.at(best_h, dec, d["y_J"])
    best_b = np.full(n, np.inf)
    np.minimum.at(best_b, dec, d["y_J_belief"])
    order = np.lexsort((np.arange(len(dec)), d["y_J"], dec))
    first = order[np.r_[True, dec[order][1:] != dec[order][:-1]]]
    out = np.full(n, np.nan)
    out[dec[first]] = d["y_J_belief"][first] - best_b[dec[first]]
    return out


def fig_dataset():
    sets = {name: load_dev(root) for name, root in SETS.items()}
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.05), gridspec_kw={"width_ratios": [1.5, 1, 1, 1.05],
                                                                      "wspace": 0.5})
    ax = axes[0]
    groups, h_share, b_share = [], [], []
    for name, d in sets.items():
        fam = d["dec_family"].astype(bool)
        oh, ah = per_decision(d, "y_J")
        ob, ab = per_decision(d, "y_J_belief")
        both = np.isfinite(oh) & np.isfinite(ah)
        subsets = [("all", both), ("fam.", both & fam)]
        if name == "stress":
            bk = d["dec_backup_available"].astype(bool)
            subsets = [("all", both), ("fam.\n+bkp", both & fam & bk), ("fam.\n" + "\u2212" + "bkp", both & fam & ~bk)]
        for label, sel in subsets:
            groups.append(label)
            h_share.append(100 * np.mean(oh[sel] < ah[sel]))
            b_share.append(100 * np.mean(ob[sel] < ab[sel]))
    x = np.arange(len(groups))
    ax.bar(x - 0.19, h_share, 0.38, color=C["hindsight"], label="hindsight labels", edgecolor="none")
    ax.bar(x + 0.19, b_share, 0.38, color=C["belief"], label="belief labels", edgecolor="none")
    for xi, v in zip(x, b_share):
        ax.text(xi + 0.19, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=6, color=C["belief"])
    ax.set_xticks(x, groups, fontsize=6.5)
    for xc, name in ((0.5, "natural"), (3.0, "stress")):
        ax.annotate(name, xy=(xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -20),
                    textcoords="offset points", ha="center", va="top", fontsize=7.5, style="italic")
    ax.axvline(1.5, ymax=0.7, color="#888888", lw=0.4, ls=":")
    ax.set_ylabel("gathering optimal (% of decisions)")
    ax.set_ylim(0, max(b_share) * 1.22)
    ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.1)
    panel(ax, "a", "gathering optimal")
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    for name, d in sets.items():
        fam = d["dec_family"].astype(bool)
        p = d["dec_posterior"][fam]
        ax.hist(p, bins=bins, histtype="step", density=False, weights=np.full(p.size, 100.0 / p.size),
                color=C[name], lw=1.1, label=name)
    ax.set_xlabel(r"$P(c^\star \mid \mathcal{E}_\tau)$, true cause")
    ax.set_ylabel("ambiguous decisions (%)")
    ax.legend(loc="upper center", handlelength=1.2)
    panel(ax, "b", "posterior")
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=C["ref"], lw=0.6, ls=":")
    for name, d in sets.items():
        cause = family_cause(d)
        fam = cause != 0
        p_link = np.where(cause[fam] > 0, d["dec_posterior"][fam], 1 - d["dec_posterior"][fam])
        y = (cause[fam] > 0).astype(float)
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p_link, edges) - 1, 0, 9)
        mp, fr, nn = [], [], []
        for b in range(10):
            s = idx == b
            if s.sum() >= 10:
                mp.append(p_link[s].mean())
                fr.append(y[s].mean())
                nn.append(s.sum())
        ax.scatter(mp, fr, s=4 + 18 * np.sqrt(np.array(nn) / max(nn)), color=C[name], zorder=3, label=name,
                   edgecolor="white", linewidth=0.3)
        ax.plot(mp, fr, color=C[name], lw=0.6, alpha=0.6)
    ax.set_xlabel(r"$P(\mathrm{link} \mid \mathcal{E}_\tau)$")
    ax.set_ylabel("observed link-fault rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", handlelength=1.0, markerscale=0.8)
    panel(ax, "c", "calibration")
    ax = axes[3]
    for name, d in sets.items():
        fam = d["dec_family"].astype(bool)
        r = belief_regret_of_hindsight_choice(d)[fam]
        r = np.sort(r[np.isfinite(r)])
        thr = np.logspace(-2, 2, 200)
        ccdf = [100 * np.mean(r > t) for t in thr]
        ax.plot(thr, ccdf, color=C[name], lw=1.1, label=name)
        print(f"belief regret of the hindsight-best action, {name} family decisions: mean {r.mean():.3f}")
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel("threshold (cost points)")
    ax.set_ylabel("ambiguous decisions above (%)")
    ax.set_ylim(0, 40)
    ax.legend(loc="upper right", handlelength=1.2)
    panel(ax, "d", "hindsight-best action")
    save(fig, "fig_dataset")

NAMES = {"playbook_act_first": "playbook: act first", "playbook_investigate_first": "playbook: investigate first",
         "monitor": "always wait (monitor 300 s)", "random": "random candidate",
         "action_prior_belief": "action prior (evidence-blind)", "base": "direct: item path",
         "base_ensemble": "direct: item path, ens.", "summary_items": "direct: summary + items",
         "summary_aux": "direct: summary as aux. target", "summary": "direct: summary only",
         "summary_ensemble": "direct: summary only, ens.", "flat_gbm_hindsight": "GBM, hindsight labels",
         "flat_gbm_belief": "GBM, belief labels", "hindsight_oracle": "hindsight oracle (reference)"}


def numbers():
    return json.loads((ROOT / "paper" / "numbers.json").read_text(encoding="utf-8"))


def fig_results():
    n = numbers()
    order = ["random", "playbook_act_first", "monitor", "playbook_investigate_first", "action_prior_belief", "base",
             "base_ensemble", "summary_aux", "summary_items", "summary", "summary_ensemble", "flat_gbm_hindsight",
             "flat_gbm_belief", "hindsight_oracle"]
    colour = {m: (C["gbm"] if m.startswith("flat") else C["net"] if m.startswith(("base", "summary"))
                  else C["ref"]) for m in order}
    colour["flat_gbm_hindsight"] = "#8fbf8f"
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.0), sharey=True, gridspec_kw={"wspace": 0.06})
    y = np.arange(len(order))[::-1]
    for ax, (set_name, letter) in zip(axes, (("natural", "a"), ("stress", "b"))):
        sc = n[f"{set_name}_main"]["test"]["belief/all"]["scores"]
        val = n[f"{set_name}_main"]["val"]["belief/all"]["scores"]
        ref = sc["flat_gbm_belief"]["regret"]["point"]
        ax.axvspan(sc["flat_gbm_belief"]["regret"]["lo"], sc["flat_gbm_belief"]["regret"]["hi"], color=C["gbm"],
                   alpha=0.07, lw=0)
        ax.axvline(ref, color=C["gbm"], lw=0.6, ls=":")
        for yi, m in zip(y, order):
            r = sc[m]["regret"]
            mk = "D" if m == "hindsight_oracle" else "o"
            face = "white" if m == "hindsight_oracle" else colour[m]
            ax.errorbar(r["point"], yi, xerr=[[r["point"] - r["lo"]], [r["hi"] - r["point"]]], fmt=mk, ms=3.6,
                        color=colour[m], mfc=face, mew=0.8, elinewidth=0.9, capsize=0)
            ax.plot(val[m]["regret"]["point"], yi + 0.28, marker="|", ms=4, color=colour[m], alpha=0.55, lw=0)
        ax.set_xlim(0, None)
        ax.set_xlabel("regret, belief labels (cost points)")
        panel(ax, letter, f"{set_name} set, test ({n[f'{set_name}_main']['test']['belief/all']['decisions']} "
                          f"decisions, {n[f'{set_name}_main']['test']['belief/all']['episodes']} episodes)")
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [NAMES[m] for m in order])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    save(fig, "fig_results")


def fig_labels():
    n = numbers()
    rows = [("natural", "all", "all decisions"), ("natural", "family", "ambiguous family"),
            ("stress", "all", "all decisions"), ("stress", "family", "ambiguous family"),
            ("stress", "backup_available", "backup channel available"),
            ("stress", "backup_unavailable", "no backup channel"),
            ("stress", "family_backup_unavailable", "ambiguous family, no backup")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True, gridspec_kw={"wspace": 0.06})
    y = np.arange(len(rows))[::-1]
    for ax, (labels, letter, title) in zip(axes, (("belief", "a", "scored with belief labels"),
                                                  ("hindsight", "b", "scored with hindsight labels (no posterior)"))):
        ax.axvline(0, color="black", lw=0.6)
        for yi, (set_name, sub, _) in zip(y, rows):
            for split, dy, filled in (("val", 0.17, False), ("test", -0.17, True)):
                d = n[f"{set_name}_main"][split][f"{labels}/{sub}"]["paired"]["flat_gbm_hindsight"]["regret"]
                col = C[set_name]
                sig = d["lo"] > 0 or d["hi"] < 0
                ax.errorbar(d["point"], yi + dy, xerr=[[d["point"] - d["lo"]], [d["hi"] - d["point"]]], fmt="o",
                            ms=3.6, color=col, mfc=col if filled else "white", mew=0.8, elinewidth=0.9)
                if sig:
                    ax.text(d["hi"] + 0.04, yi + dy, "*", va="center", ha="left", fontsize=8, color=col)
        ax.set_xlabel(r"regret difference ($<0$: belief labels better)")
        panel(ax, letter, title)
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [f"{s}: {t}" for s, _, t in rows])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", ls="", color="#555555", mfc="white", ms=4, label="validation"),
               plt.Line2D([], [], marker="o", ls="", color="#555555", ms=4, label="test (frozen, one look)")]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    save(fig, "fig_labels")


def fig_sensitivity():
    n = numbers()
    Ts, deltas = ["0.0", "0.5", "1.0", "2.0", "inf"], [-1.0, 0.0, 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.05), gridspec_kw={"wspace": 0.12})
    vmax = 1.0
    cmap = plt.get_cmap("RdBu_r")
    k = 0
    for set_name in ("natural", "stress"):
        sens = n[f"{set_name}_sensitivity"]
        by = {(str(float(c["T"])) if c["T"] != "inf" else "inf", float(c["delta"])): c for c in sens["configs"]}
        for labels in ("belief", "hindsight"):
            ax = axes[k]
            M = np.full((len(Ts), len(deltas)), np.nan)
            for i, T in enumerate(Ts):
                for j, dl in enumerate(deltas):
                    d = by[(T, dl)]["splits"]["test"]["views"][f"{labels}/all"]["minus_gbm_hindsight"]
                    M[i, j] = d["point"]
                    sig = d["lo"] > 0 or d["hi"] < 0
                    ax.text(j, i, f"{d['point']:+.2f}" + ("*" if sig else ""), ha="center", va="center",
                            fontsize=6.6, color="white" if abs(d["point"]) > 0.55 * vmax else "black")
            ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(3), [r"$-1$", r"$0$", r"$+1$"])
            ax.set_xlabel(r"prior shift $\delta$")
            ax.set_yticks(range(len(Ts)), ["0 (prior)", "0.5", "1 (exact)", "2", r"$\infty$ (hard)"] if k == 0 else [])
            if k == 0:
                ax.set_ylabel(r"evidence weight $T$")
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            ax.add_patch(plt.Rectangle((0.5, 1.5), 1, 1, fill=False, ec="black", lw=0.8))
            panel(ax, "abcd"[k], f"{set_name}, {labels}")
            k += 1
    save(fig, "fig_sensitivity")


def fig_curves():
    n = numbers()
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.05), gridspec_kw={"wspace": 0.55})
    fr = ["0.25", "0.5", "1.0"]
    x = [25, 50, 100]
    for ax, set_name, letter in ((axes[0], "natural", "a"), (axes[1], "stress", "b")):
        lc = n["learning_curves"][set_name]
        for key, col, lab, mk in (("base", C["base"], "direct: item path", "s"),
                                  ("summary", C["net"], "direct: summary only", "o"),
                                  ("flat_gbm", C["gbm"], "GBM", "^")):
            ax.plot(x, [lc[f][key] for f in fr], marker=mk, ms=3.5, color=col, lw=1.0, label=lab)
        ax.set_xticks(x, ["25%", "50%", "100%"])
        ax.set_xlabel("share of train episodes")
        ax.set_ylabel("val regret (belief)")
        ax.set_ylim(0, None)
        panel(ax, letter, f"learning, {set_name}")
    axes[0].legend(loc="lower left", handlelength=1.4, fontsize=6.6)
    methods = [("base", "item\npath", C["base"]), ("summary", "summary\nonly", C["net"]),
               ("flat_gbm_belief", "GBM", C["gbm"])]
    for ax, key, letter, title, ylab in (
            (axes[2], "tail_gather_share", "c", "costly gathers (test)", "gathers when gap $\\geq$ 10 (%)"),
            (axes[3], "act_corr_high_stakes", "d", "act ranking (test)", "corr. of act costs, high stakes")):
        w = 0.36
        for j, set_name in enumerate(("natural", "stress")):
            dg = n[f"{set_name}_main"]["test"]["belief/all"]["diagnostics"]
            vals = [dg[m][key] * (100 if key == "tail_gather_share" else 1) for m, _, _ in methods]
            xs = np.arange(len(methods)) + (j - 0.5) * w
            ax.bar(xs, vals, w, color=[c for _, _, c in methods], alpha=1.0 if set_name == "stress" else 0.45,
                   edgecolor="none", label=set_name)
        ax.set_xticks(np.arange(len(methods)), [lab for _, lab, _ in methods], fontsize=6.6)
        ax.set_ylabel(ylab, fontsize=7.2)
        panel(ax, letter, title)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.45, label="natural"),
               plt.Rectangle((0, 0), 1, 1, color="#777777", label="stress")]
    axes[3].set_ylim(0, 0.62)
    axes[2].set_ylim(0, 112)
    axes[3].legend(handles=handles, loc="upper left", fontsize=6.6, handlelength=1.0, ncol=2, columnspacing=0.8)
    save(fig, "fig_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("dataset", "results", "sensitivity", "curves"))
    parser.add_argument("--natural")
    parser.add_argument("--stress")
    parser.add_argument("--labels-natural")
    parser.add_argument("--labels-stress")
    args = parser.parse_args()
    style()
    if args.what == "dataset":
        fig_dataset()
    elif args.what == "results":
        fig_results()
        fig_labels()
    elif args.what == "sensitivity":
        fig_sensitivity()
    elif args.what == "curves":
        fig_curves()

if __name__ == "__main__":
    main()
