"""E09-E12: where the advantage of noise-augmented training comes from, and whether
it is complementary to calibrated averaging. Open-loop, from frozen predictions.

python analysis_aug.py --platform sim --data_dir ../data_sim --runs ../runs/sim \
       --frozen ../runs/sim/frozen.json --out ../results/sim

E09 (crossing)   dictionaries of augmented members (D-MA, D-SA; on the arm also the
                 channel-matched D-MQ, D-SQ), mixed dictionaries (D-M+A = D-M plus the
                 B-A network, D-U = D-M united with D-MA, D-US = D-S united with D-SA),
                 the same rules as E01 (W-S, W-E, W-Q2, W-Q5) at the full calibration
                 budget, with the same execution budget k <= 5.
E10 (mechanism)  split of every member residual under a perturbed condition into the
                 propagated-perturbation term n_j = f_j(x_tilde) - f_j(x) and the nominal
                 term b_j = f_j(x) - y; second moments and correlations of the two terms
                 across members; calibration slope of every member; distance from the
                 hull of a dictionary to the augmented network.
E11 (front end)  the frozen clean members behind the label-free linear input estimator
                 written by frontend_preds.py (suffix wfe: estimated noise variances,
                 wfo: oracle variances), under the same rules.
E12 (controls)   D-M without the STE member; dictionaries extended by the null
                 predictor (the output mean), which turns the simplex into a set that
                 can shrink the output gain.

Outputs: e09_rows.csv, e09_summary.csv, e09_paired.csv, e09_weights.csv, e09_units.npz,
         e10_members.csv, e10_dicts.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.surrogates import SURROGATE_NAMES
from common import calibration as cal
from common.metrics import nmse, nmse_rows, r2_pooled, cluster_bootstrap


def cname(c):
    return c.replace(":", "_")


class Runs:
    def __init__(self, runs_dir, strong):
        self.dir, self.strong, self._c = runs_dir, strong, {}

    def members(self, block, seed):
        s = self.strong
        if block in ("dm", "dma", "dmq"):
            return [f"{block}_{x}_s{seed}" for x in SURROGATE_NAMES]
        if block in ("ds", "dsa", "dsq"):
            return [f"{block}_{s}_s{seed}_m{i}" for i in range(5)]
        if block == "ba":
            return [f"ba_{s}_s{seed}"]
        if block in ("bn", "bna", "bnq", "bnas", "bnqs"):
            return [f"{block}_mlp_s{seed}"]
        if block in ("dmas", "dmqs"):          # only the strong-surrogate member is trained (B-A+)
            return [f"{block}_{s}_s{seed}"]
        raise ValueError(block)

    def path(self, mid, role, cond, var=""):
        return os.path.join(self.dir, mid, f"pred_{role}_{cname(cond)}{('__' + var) if var else ''}.npy")

    def has(self, mids, role, cond, var=""):
        return all(os.path.exists(self.path(m, role, cond, var)) for m in mids)

    def pred(self, mid, role, cond, var=""):
        key = (mid, role, cond, var)
        if key not in self._c:
            self._c[key] = np.load(self.path(mid, role, cond, var)).astype(np.float64)
        return self._c[key]


def simplex_min(C):
    return cal.budgeted_qp(C, C.shape[0])


def moment(A, B, d):
    """[M,N,d] x [M',N,d] -> [M,M'] matrix of E[a_i . b_j] / d."""
    return np.einsum("ind,jnd->ij", A, B) / (A.shape[1] * d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n_boot", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frozen = json.load(open(args.frozen))
    seeds = [int(x) for x in args.seeds.split(",")]
    sim = args.platform == "sim"
    if sim:
        from sim.data import load_sim
        from sim.panda_env import TORQUE_LIMIT
        D = load_sim(args.data_dir)
        lo = (-TORQUE_LIMIT - D["y_mean"]) / D["y_std"]; hi = (TORQUE_LIMIT - D["y_mean"]) / D["y_std"]
    else:
        from sarcos.data import load_sarcos
        D = load_sarcos(args.data_dir)
        lo = hi = None
    clip = (lambda P: np.clip(P, lo, hi)) if lo is not None else (lambda P: P)
    R = Runs(args.runs, frozen["strong_surrogate"])
    d = D["n_out"]; scale = np.ones(d)
    istrong = SURROGATE_NAMES.index(R.strong)
    cases = [("cal", "test", c) for c in frozen["conds"]]
    if sim:
        cases.append(("cal_P1", "test_P1", "N"))

    rows, wrows, units = [], [], {}
    mem_rows, dict_rows = [], []
    for seed in seeds:
        for cal_role, test_role, cond in cases:
            lab = "P1" if cal_role.endswith("_P1") else cond
            Ycal = D["Y"][cal_role].astype(np.float64); Ytest = D["Y"][test_role].astype(np.float64)
            gcal = D["groups"][cal_role] if sim else None
            gtest = D["groups"][test_role] if sim else None

            def ok(block, var=""):
                m = R.members(block, seed)
                return R.has(m, cal_role, cond, var) and R.has(m, test_role, cond, var)

            def PP(block, var="", keep=None):
                m = R.members(block, seed)
                if keep is not None:
                    m = [m[i] for i in keep]
                return {"cal": [clip(R.pred(x, cal_role, cond, var)) for x in m],
                        "test": [clip(R.pred(x, test_role, cond, var)) for x in m]}

            def join(*ps):
                return {"cal": sum((p["cal"] for p in ps), []), "test": sum((p["test"] for p in ps), [])}

            zero = {"cal": [np.zeros_like(Ycal)], "test": [np.zeros_like(Ytest)]}
            S = list(SURROGATE_NAMES)
            seedlab = [f"seed{i}" for i in range(5)]
            dicts = {}
            if not ok("dm"):
                print("missing D-M", seed, lab); continue
            dicts["D-M"] = (PP("dm"), S)
            no_ste = [i for i, x in enumerate(S) if x != "ste"]
            dicts["D-M-STE"] = (PP("dm", keep=no_ste), [S[i] for i in no_ste])
            dicts["D-M+0"] = (join(PP("dm"), zero), S + ["null"])
            if ok("ds"):
                dicts["D-S"] = (PP("ds"), seedlab)
                dicts["D-S+0"] = (join(PP("ds"), zero), seedlab + ["null"])
            if ok("dmas"):
                dicts["D-M+A+"] = (join(PP("dm"), PP("dmas")), S + ["B-A+"])
                if ok("ds"):
                    dicts["D-S+A+"] = (join(PP("ds"), PP("dmas")), seedlab + ["B-A+"])
            if ok("dmqs"):
                dicts["D-M+AQ+"] = (join(PP("dm"), PP("dmqs")), S + ["B-AQ+"])
                if ok("ds"):
                    dicts["D-S+AQ+"] = (join(PP("ds"), PP("dmqs")), seedlab + ["B-AQ+"])
            if ok("ba"):
                dicts["D-M+A"] = (join(PP("dm"), PP("ba")), S + ["B-A"])
                if ok("ds"):
                    dicts["D-S+A"] = (join(PP("ds"), PP("ba")), seedlab + ["B-A"])
            if ok("dma"):
                dicts["D-MA"] = (PP("dma"), [x + "+aug" for x in S])
                dicts["D-MA+0"] = (join(PP("dma"), zero), [x + "+aug" for x in S] + ["null"])
                dicts["D-U"] = (join(PP("dm"), PP("dma")), S + [x + "+aug" for x in S])
            if ok("dsa"):
                dicts["D-SA"] = (PP("dsa"), [x + "+aug" for x in seedlab])
                if ok("ds"):
                    dicts["D-US"] = (join(PP("ds"), PP("dsa")), seedlab + [x + "+aug" for x in seedlab])
            if ok("dmq"):
                dicts["D-MQ"] = (PP("dmq"), [x + "+augq" for x in S])
                dicts["D-UQ"] = (join(PP("dm"), PP("dmq")), S + [x + "+augq" for x in S])
            if ok("dsq"):
                dicts["D-SQ"] = (PP("dsq"), [x + "+augq" for x in seedlab])
            for var in ("wfe", "wfo"):
                if ok("dm", var):
                    dicts[f"D-M|{var}"] = (PP("dm", var), S)
                    dicts[f"D-M|{var}+0"] = (join(PP("dm", var), zero), S + ["null"])
                if ok("ds", var):
                    dicts[f"D-S|{var}"] = (PP("ds", var), seedlab)
                    dicts[f"D-S|{var}+0"] = (join(PP("ds", var), zero), seedlab + ["null"])
                if ok("dma", var):
                    dicts[f"D-MA|{var}"] = (PP("dma", var), [x + "+aug" for x in S])
            # raw members united with their front-end versions: the calibration decides
            # per condition whether the front end is used (10 candidates, k <= 5 executed)
            if ok("dm", "wfe"):
                dicts["D-MW"] = (join(PP("dm"), PP("dm", "wfe")), S + [x + "|wfe" for x in S])
            if ok("ds", "wfe"):
                dicts["D-SW"] = (join(PP("ds"), PP("ds", "wfe")), seedlab + [x + "|wfe" for x in seedlab])
            singles = {"B-F": PP("dm", keep=[istrong])["test"][0]}
            for name, block, var, keep in [("B-A", "ba", "", None), ("B-N", "bn", "", None), ("B-NA", "bna", "", None),
                                           ("B-AQ", "dmq", "", [istrong]), ("B-NQ", "bnq", "", None),
                                           ("B-A+", "dmas", "", None), ("B-AQ+", "dmqs", "", None),
                                           ("B-NA+", "bnas", "", None), ("B-NQ+", "bnqs", "", None),
                                           ("B-A+|wfe", "dmas", "wfe", None), ("B-AQ+|wfe", "dmqs", "wfe", None),
                                           ("B-F|wfe", "dm", "wfe", [istrong]), ("B-F|wfo", "dm", "wfo", [istrong]),
                                           ("B-A|wfe", "ba", "wfe", None), ("B-A|wfo", "ba", "wfo", None),
                                           ("B-N|wfe", "bn", "wfe", None), ("B-N|wfo", "bn", "wfo", None)]:
                if ok(block, var):
                    singles[name] = PP(block, var, keep=keep)["test"][0]

            def add(method, dn, P, w=None, labels=None):
                r = nmse_rows(P, Ytest, scale)
                u = pd.Series(r).groupby(gtest).mean().values if gtest is not None else r
                rows.append({"seed": seed, "cond": lab, "method": method, "dictionary": dn, "nmse": float(r.mean()),
                             "r2": r2_pooled(P, Ytest), "support_size": int(np.sum(np.asarray(w) > 1e-12)) if w is not None else 1})
                units[f"{seed}|{lab}|{method}|{dn}"] = u
                if w is not None and labels is not None:
                    for l, x in zip(labels, w):
                        wrows.append({"seed": seed, "cond": lab, "dictionary": dn, "rule": method, "member": l, "weight": float(x)})

            for name, P in singles.items():
                add(name, "-", P)
            for dn, (P, labels) in dicts.items():
                M = len(P["cal"])
                Chat, _, _ = cal.residual_matrix(P["cal"], Ycal, scale, gcal)
                Ctest, _, _ = cal.residual_matrix(P["test"], Ytest, scale, gtest)
                rule = {"W-S": cal.rule_WS(Chat), "W-E": cal.rule_WE(Chat),
                        "W-Q2": cal.budgeted_qp(Chat, min(2, M))["w"], "W-Q5": cal.budgeted_qp(Chat, min(5, M))["w"],
                        "posthoc-opt": cal.budgeted_qp(Ctest, min(5, M))["w"]}
                for rn, w in rule.items():
                    add(rn, dn, sum(wj * p for wj, p in zip(w, P["test"])), w, labels)
                for j, l in enumerate(labels):
                    rows.append({"seed": seed, "cond": lab, "method": f"member:{l}", "dictionary": dn,
                                 "nmse": float(Ctest[j, j]), "r2": np.nan, "support_size": 1})

            # ---------------- E10: propagated-perturbation / nominal split on the test role
            if lab not in ("N", "P1"):
                A_ref = singles.get("B-A")
                for dn, block, labels, var in [("D-M", "dm", S, ""), ("D-S", "ds", seedlab, ""), ("D-MA", "dma", S, ""), ("D-SA", "dsa", seedlab, ""),
                                               ("D-MQ", "dmq", S, ""), ("D-SQ", "dsq", seedlab, ""), ("B-A", "ba", ["B-A"], ""),
                                               ("B-N", "bn", ["B-N"], ""), ("B-NA", "bna", ["B-NA"], ""), ("B-NQ", "bnq", ["B-NQ"], ""),
                                               ("B-A+", "dmas", ["B-A+"], ""), ("B-AQ+", "dmqs", ["B-AQ+"], ""),
                                               ("B-NA+", "bnas", ["B-NA+"], ""), ("B-NQ+", "bnqs", ["B-NQ+"], ""),
                                               ("D-M|wfe", "dm", S, "wfe"), ("D-S|wfe", "ds", seedlab, "wfe"),
                                               ("D-M|wfo", "dm", S, "wfo"), ("D-S|wfo", "ds", seedlab, "wfo")]:
                    mids = R.members(block, seed)
                    if not (R.has(mids, test_role, cond, var) and R.has(mids, test_role, "N")):
                        continue
                    Pt = np.stack([clip(R.pred(m, test_role, cond, var)) for m in mids])  # perturbed inputs (behind the front end if var)
                    Pc = np.stack([clip(R.pred(m, test_role, "N")) for m in mids])       # clean inputs, same rows
                    n = Pt - Pc; b = Pc - Ytest[None]
                    Cnn, Cbb, Cnb = moment(n, n, d), moment(b, b, d), moment(n, b, d)
                    for j, l in enumerate(labels):
                        beta = float(np.sum(Pt[j] * Ytest) / np.sum(Pt[j] ** 2))
                        mem_rows.append({"seed": seed, "cond": lab, "dictionary": dn, "member": l,
                                         "risk": float(nmse(Pt[j], Ytest, scale)), "P_n": float(Cnn[j, j]), "P_b": float(Cbb[j, j]),
                                         "cross": float(2 * Cnb[j, j]), "slope": beta,
                                         "risk_shrunk": float(nmse(beta * Pt[j], Ytest, scale)),
                                         "output_power": float(np.mean(Pt[j] ** 2)), "target_power": float(np.mean(Ytest ** 2))})
                    M = len(mids)
                    if M > 1:
                        off = ~np.eye(M, dtype=bool)
                        corr = lambda C: float(np.mean((C / np.sqrt(np.outer(np.diag(C), np.diag(C))))[off]))
                        Ctot, _, _ = cal.residual_matrix(list(Pt), Ytest, scale, gtest)
                        q = simplex_min(Ctot)
                        ent = {"seed": seed, "cond": lab, "dictionary": dn,
                               "mean_P_n": float(np.mean(np.diag(Cnn))), "mean_P_b": float(np.mean(np.diag(Cbb))),
                               "corr_n": corr(Cnn), "corr_b": corr(Cbb),
                               "common_mode_n": float(Cnn.sum() / M ** 2 / np.mean(np.diag(Cnn))),
                               "simplex_floor_n": simplex_min(Cnn)["value"], "simplex_floor_b": simplex_min(Cbb)["value"],
                               "simplex_floor_total": q["value"], "best_member_total": float(np.min(np.diag(Ctot))),
                               "P_n_at_wstar": float(q["w"] @ Cnn @ q["w"]), "P_b_at_wstar": float(q["w"] @ Cbb @ q["w"]),
                               "cross_at_wstar": float(2 * q["w"] @ Cnb @ q["w"])}
                        if A_ref is not None and dn != "B-A":
                            Dm = Pt - A_ref[None]
                            G = moment(Dm, Dm, d)
                            qa = simplex_min(G)
                            rA = (A_ref - Ytest)[None]
                            ent.update({"risk_BA": float(nmse(A_ref, Ytest, scale)),
                                        "hull_dist2_to_BA": qa["value"],
                                        "dist2_wstar_to_BA": float(q["w"] @ G @ q["w"]),
                                        "cross_wstar_BA": float(2 * (q["w"] @ moment(Dm, rA, d))[0])})
                        dict_rows.append(ent)
        print("seed", seed, "done", flush=True)

    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.out, "e09_rows.csv"), index=False)
    pd.DataFrame(wrows).to_csv(os.path.join(args.out, "e09_weights.csv"), index=False)
    pd.DataFrame(mem_rows).to_csv(os.path.join(args.out, "e10_members.csv"), index=False)
    pd.DataFrame(dict_rows).to_csv(os.path.join(args.out, "e10_dicts.csv"), index=False)
    np.savez_compressed(os.path.join(args.out, "e09_units.npz"), **units)

    summ = []
    for (lab, method, dn), sub in df.groupby(["cond", "method", "dictionary"]):
        if method.startswith("member:"):
            summ.append({"cond": lab, "method": method, "dictionary": dn, "nmse_mean": sub["nmse"].mean(), "ci_lo": np.nan,
                         "ci_hi": np.nan, "r2_mean": np.nan, "n_seeds": len(sub), "seed_std": sub["nmse"].std(ddof=1)})
            continue
        vals, cs, cu = [], [], []
        for s in sub["seed"].unique():
            u = units[f"{int(s)}|{lab}|{method}|{dn}"]
            vals.append(u); cs.append(np.full(len(u), s)); cu.append(np.arange(len(u)))
        v = np.concatenate(vals)
        pt, l, h = cluster_bootstrap(v, {"seed": np.concatenate(cs), "unit": np.concatenate(cu)}, n_boot=args.n_boot)
        summ.append({"cond": lab, "method": method, "dictionary": dn, "nmse_mean": pt, "ci_lo": l, "ci_hi": h,
                     "r2_mean": sub["r2"].mean(), "n_seeds": len(vals), "seed_std": sub["nmse"].std(ddof=1) if len(sub) > 1 else np.nan})
    pd.DataFrame(summ).to_csv(os.path.join(args.out, "e09_summary.csv"), index=False)

    PAIRS = [("W-Q5|D-MA", "B-A|-"), ("W-Q5|D-MA", "W-Q5|D-SA"), ("W-Q5|D-MA", "W-S|D-MA"), ("W-Q5|D-SA", "B-A|-"),
             ("W-Q5|D-M+A", "B-A|-"), ("W-Q5|D-M+A", "W-Q5|D-M"), ("W-Q5|D-M+A", "W-Q5|D-S+A"),
             ("W-Q5|D-U", "B-A|-"), ("W-Q5|D-U", "W-Q5|D-M"), ("W-Q5|D-U", "W-Q5|D-MA"), ("W-Q5|D-U", "W-Q5|D-US"), ("W-Q2|D-U", "B-A|-"),
             ("W-Q5|D-M+0", "W-Q5|D-M"), ("W-Q5|D-S+0", "W-Q5|D-S"), ("W-Q5|D-M+0", "W-Q5|D-S+0"), ("W-Q5|D-MA+0", "W-Q5|D-MA"),
             ("W-Q5|D-M-STE", "W-Q5|D-S"), ("W-Q5|D-M-STE", "W-Q5|D-M"),
             ("W-Q5|D-M|wfe", "W-Q5|D-M"), ("W-Q5|D-M|wfe", "B-A|-"), ("W-Q5|D-M|wfe", "W-Q5|D-S|wfe"), ("W-Q5|D-M|wfe", "B-F|wfe|-"),
             ("W-Q5|D-M|wfo", "W-Q5|D-M"), ("W-Q5|D-M|wfo", "B-A|-"), ("W-Q5|D-M|wfo", "W-Q5|D-S|wfo"),
             ("W-Q5|D-M", "W-Q5|D-S"), ("W-Q5|D-M", "B-A|-"), ("W-Q5|D-S", "B-A|-"),
             ("W-Q5|D-M|wfe+0", "W-Q5|D-S|wfe+0"), ("W-Q5|D-M|wfe+0", "W-Q5|D-M|wfe"), ("W-Q5|D-S|wfe+0", "W-Q5|D-S|wfe"),
             ("W-Q5|D-M|wfo+0", "W-Q5|D-S|wfo+0"), ("W-Q5|D-S|wfe", "B-A|-"), ("W-Q5|D-S|wfe", "W-Q5|D-S"),
             ("W-Q5|D-MW", "W-Q5|D-M"), ("W-Q5|D-MW", "W-Q5|D-M|wfe"), ("W-Q5|D-MW", "W-Q5|D-SW"), ("W-Q5|D-MW", "B-A|-"),
             ("W-Q5|D-SW", "W-Q5|D-S"), ("W-Q5|D-SW", "B-A|-"), ("W-Q5|D-S+0", "B-A|-"), ("W-Q5|D-M+0", "B-A|-"),
             ("W-Q5|D-MA", "W-Q5|D-M"), ("W-Q5|D-MA", "W-Q5|D-M|wfe"), ("W-Q5|D-SA", "W-Q5|D-S"), ("W-S|D-MA", "B-A|-"),
             ("W-Q5|D-MA|wfe", "B-A|-"), ("W-Q5|D-MQ", "W-Q5|D-M|wfe"), ("W-Q5|D-MQ", "W-Q5|D-M"), ("W-Q5|D-SQ", "W-Q5|D-S"),
             ("W-Q5|D-MQ", "B-A|-"), ("W-Q5|D-SQ", "B-AQ|-"),
             ("B-A+|-", "B-A|-"), ("B-AQ+|-", "B-AQ|-"), ("B-AQ+|-", "B-A+|-"), ("B-NA+|-", "B-A+|-"), ("B-NQ+|-", "B-AQ+|-"),
             ("W-Q5|D-M", "B-A+|-"), ("W-Q5|D-M", "B-AQ+|-"), ("W-Q5|D-M|wfe", "B-A+|-"), ("W-Q5|D-M|wfe", "B-AQ+|-"),
             ("W-Q5|D-S|wfe", "B-A+|-"), ("W-Q5|D-S|wfe", "B-AQ+|-"), ("W-Q5|D-S+0", "B-A+|-"), ("W-Q5|D-S+0", "B-AQ+|-"),
             ("W-Q5|D-M+A+", "B-A+|-"), ("W-Q5|D-M+A+", "W-Q5|D-M"), ("W-Q5|D-M+A+", "W-Q5|D-S+A+"),
             ("W-Q5|D-M+AQ+", "B-AQ+|-"), ("W-Q5|D-M+AQ+", "W-Q5|D-M"), ("W-Q5|D-M+AQ+", "W-Q5|D-S+AQ+"),
             ("B-A+|wfe|-", "B-A+|-"), ("B-AQ+|wfe|-", "B-AQ+|-"), ("W-Q5|D-MA", "B-A+|-"), ("W-Q5|D-MQ", "B-AQ+|-"), ("B-NA|-", "W-Q5|D-MA"), ("B-NQ|-", "W-Q5|D-MQ"), ("B-N|wfe|-", "W-Q5|D-M|wfe"),
             ("B-F|wfe|-", "B-F|-"), ("B-F|wfo|-", "B-F|-"), ("B-A|wfe|-", "B-A|-"), ("W-Q5|D-MA|wfe", "W-Q5|D-MA"),
             ("B-NA|-", "B-A|-"), ("B-NA|-", "B-N|-"),
             ("B-AQ|-", "B-A|-"), ("W-Q5|D-MQ", "B-AQ|-"), ("W-Q5|D-MQ", "W-Q5|D-SQ"), ("W-Q5|D-M", "B-AQ|-"),
             ("W-Q5|D-UQ", "B-AQ|-"), ("W-Q5|D-UQ", "W-Q5|D-M"), ("B-NQ|-", "B-AQ|-")]
    prs = []
    labs = list(dict.fromkeys(df["cond"]))
    for lab in labs:
        for a, b in PAIRS:
            diffs, cs, cu = [], [], []
            for s in seeds:
                ka, kb = f"{s}|{lab}|{a}", f"{s}|{lab}|{b}"
                if ka in units and kb in units:
                    dd = units[ka] - units[kb]; diffs.append(dd); cs.append(np.full(len(dd), s)); cu.append(np.arange(len(dd)))
            if not diffs:
                continue
            v = np.concatenate(diffs)
            pt, l, h = cluster_bootstrap(v, {"seed": np.concatenate(cs), "unit": np.concatenate(cu)}, n_boot=args.n_boot, seed=1)
            prs.append({"cond": lab, "a": a, "b": b, "diff": pt, "ci_lo": l, "ci_hi": h,
                        "seeds_a_better": int(sum(np.mean(x) < 0 for x in diffs)), "n_seeds": len(diffs)})
    pd.DataFrame(prs).to_csv(os.path.join(args.out, "e09_paired.csv"), index=False)
    print("done", len(rows), "rows;", len(prs), "paired rows;", len(mem_rows), "member rows")


if __name__ == "__main__":
    main()
