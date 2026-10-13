"""Open-loop analysis from frozen predictions (E01-E05, E07, theory quantities).

python analysis_openloop.py --platform sim --data_dir ../data_sim --runs ../runs/sim \
       --frozen ../runs/sim/frozen.json --out ../results/sim

Writes (all CSV/JSON, every number traceable to member ids and roles):
  e01_test_table.csv      per seed x condition x method: test nMSE, pooled R2, per-joint RMSE
  e01_summary.csv         bootstrap means / 95% CI over seeds (and groups, units)
  e02_geometry.json       C(s) on D_cal and D_test, eq.(3) decomposition, two-member witness
                          identified on D_cal and verified on D_test, forward-response
  e03_m_slice.csv         rules x m (or n_cal) x condition: test nMSE, g_cal, support
  e04_k_slice.csv         W-Q(k) on D-M / D-S / D-T: test nMSE, support size
  e05_boundary.csv        relative gain, absolute error, post-hoc candidate optimum, common residual
  e07_density.csv         independent trajectories vs dense sampling (sim only)
  theory_quantities.json  B_e^2, eps_m, vhat_k, G-hat_k (sim)
  calibration_weights.json weights used by the closed-loop runner (sim)
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.surrogates import SURROGATE_NAMES
from common import calibration as cal
from common.metrics import nmse, nmse_rows, r2_pooled, per_joint_rmse, cluster_bootstrap

SEEDS = [0, 1, 2, 3, 4]
M_SLICE_SIM = [1, 2, 5, 10]
NCAL_SLICE_SARCOS = [50, 200, 1000, 4448]
N_GROUPS = 10


def cname(c):
    return c.replace(":", "_")


class Runs:
    def __init__(self, runs_dir, frozen):
        self.dir = runs_dir
        self.frozen = frozen
        self.strong = frozen["strong_surrogate"]
        self._cache = {}

    def members(self, block, seed):
        s = self.strong
        if block == "dm":
            return [f"dm_{x}_s{seed}" for x in SURROGATE_NAMES]
        if block == "ds":
            return [f"ds_{s}_s{seed}_m{i}" for i in range(5)]
        if block == "dt":
            return [f"dt_{s}_s{seed}_c{i}" for i in range(5)]
        if block == "bl":
            return [f"bl_learnable_s{seed}"]
        if block == "ba":
            return [f"ba_{s}_s{seed}"]
        if block == "bn":
            return [f"bn_mlp_s{seed}"]
        raise ValueError(block)

    def has(self, mid):
        return os.path.exists(os.path.join(self.dir, mid, "config.json"))

    def pred(self, mid, role, cond):
        key = (mid, role, cond)
        if key not in self._cache:
            self._cache[key] = np.load(os.path.join(self.dir, mid, f"pred_{role}_{cname(cond)}.npy")).astype(np.float64)
        return self._cache[key]

    def config(self, mid):
        return json.load(open(os.path.join(self.dir, mid, "config.json")))


def clip_preds(P, lo, hi):
    return np.clip(P, lo, hi) if lo is not None else P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frozen = json.load(open(args.frozen))
    seeds = [int(x) for x in args.seeds.split(",")]
    sim = args.platform == "sim"
    if sim:
        from sim.data import load_sim
        from sim.panda_env import TORQUE_LIMIT
        D = load_sim(args.data_dir)
        # clip limits in standardized units (eq. 1 with clipped base predictors)
        clip_lo = (-TORQUE_LIMIT - D["y_mean"]) / D["y_std"]; clip_hi = (TORQUE_LIMIT - D["y_mean"]) / D["y_std"]
    else:
        from sarcos.data import load_sarcos
        D = load_sarcos(args.data_dir)
        clip_lo = clip_hi = None
    R = Runs(args.runs, frozen)
    scale = np.ones(D["n_out"])          # targets already standardized by D_tau
    d = D["n_out"]
    conds = frozen["conds"]
    O1M = f"O1:{frozen['O1_M']}"
    # roles: (cal_role, test_role, cond) triples; P1 plant appears as its own role pair
    cases = [("cal", "test", c) for c in conds]
    if sim:
        cases.append(("cal_P1", "test_P1", "N"))

    def label(cal_role, cond):
        return "P1" if cal_role.endswith("_P1") else cond

    def groups(role):
        return D["groups"][role] if sim else None

    def cal_subset(role, m_or_n, g):
        """Index set of calibration unit group g with budget m (sim: m trajectories of
        group g = trajectories [10g, 10g+m) ; sarcos: n rows drawn with group seed)."""
        if sim:
            tr = D["groups"][role]
            ids = np.arange(10 * g, 10 * g + m_or_n)
            return np.where(np.isin(tr, ids))[0]
        rng = np.random.default_rng(500 + g)
        perm = rng.permutation(len(D["Y"][role]))
        return perm[:m_or_n]

    def fit_rules(preds_cal, Ycal, gcal, k_list=(1, 2, 3, 4, 5)):
        """All weight rules on one calibration set. Returns dict name -> (w, info)."""
        Chat, A, _ = cal.residual_matrix(preds_cal, Ycal, scale, gcal)
        mse_list = [nmse(p, Ycal, scale) for p in preds_cal]
        n_scalar = Ycal.size
        out = {"W-S": (cal.rule_WS(Chat), {}), "W-E": (cal.rule_WE(Chat), {}),
               "W-B": (cal.rule_WB(mse_list, n_scalar), {})}
        wd, nm = cal.rule_WD(Chat, mse_list, n_scalar); out["W-D"] = (wd, {"choice": nm})
        for k in k_list:
            q = cal.budgeted_qp(Chat, k)
            out[f"W-Q{k}"] = (q["w"], {"support": q["support"], "kkt": q["kkt_residual"], "vhat_k": q["value"]})
        for name, (w, info) in out.items():
            info["risk_cal"] = cal.risk(w, Chat)
            info["g_cal_5"] = info["risk_cal"] - out["W-Q5"][1]["vhat_k"]
            info["support_size"] = int(np.sum(w > 0))
        return out, Chat

    def test_eval(w, preds_test, Ytest, gtest):
        P = sum(wj * p for wj, p in zip(w, preds_test))
        rows = nmse_rows(P, Ytest, scale)
        return {"nmse": float(rows.mean()), "r2": r2_pooled(P, Ytest),
                "rmse_joint": per_joint_rmse(P * D["y_std"], Ytest * D["y_std"]).tolist(),
                "unit_nmse": (pd.Series(rows).groupby(gtest).mean().values if gtest is not None else rows)}

    e01_rows, e01_units = [], {}
    e03_rows, e04_rows, e05_rows, e07_rows = [], [], [], []
    geometry = {}
    weights_out = {}
    theory = {}
    for seed in seeds:
        if not all(R.has(m) for m in R.members("dm", seed)):
            print("seed", seed, "incomplete, skipping"); continue
        have = {b: all(R.has(m) for m in R.members(b, seed)) for b in ["ds", "dt", "bl", "ba", "bn"]}
        for cal_role, test_role, cond in cases:
            lab = label(cal_role, cond)
            Ycal, Ytest = D["Y"][cal_role], D["Y"][test_role]
            gcal, gtest = groups(cal_role), groups(test_role)
            dicts = {"D-M": R.members("dm", seed)}
            if have["ds"]: dicts["D-S"] = R.members("ds", seed)
            if have["dt"]: dicts["D-T"] = R.members("dt", seed)
            P = {}
            for name, mids in dicts.items():
                P[name] = {"cal": [clip_preds(R.pred(m, cal_role, cond), clip_lo, clip_hi) for m in mids],
                           "test": [clip_preds(R.pred(m, test_role, cond), clip_lo, clip_hi) for m in mids]}
            singles = {}
            for b in ["bl", "ba", "bn"]:
                if have[b]:
                    m = R.members(b, seed)[0]
                    singles[b] = clip_preds(R.pred(m, test_role, cond), clip_lo, clip_hi)
            # --- E01: singles and dictionaries under common rules (full calibration budget)
            def add_row(method, dictionary, w, te, extra=None):
                row = {"seed": seed, "cond": lab, "method": method, "dictionary": dictionary,
                       "nmse": te["nmse"], "r2": te["r2"], "support_size": int(np.sum(np.asarray(w) > 0)) if w is not None else 1}
                for j, v in enumerate(te["rmse_joint"]):
                    row[f"rmse_j{j + 1}"] = v
                if extra: row.update(extra)
                e01_rows.append(row)
                e01_units[(seed, lab, method, dictionary)] = te["unit_nmse"]
            for i, x in enumerate(SURROGATE_NAMES):
                w = np.eye(5)[i]
                add_row(f"single:{x}", "D-M", w, test_eval(w, P["D-M"]["test"], Ytest, gtest))
            w = np.eye(5)[SURROGATE_NAMES.index(R.strong)]
            add_row("B-F", "D-M", w, test_eval(w, P["D-M"]["test"], Ytest, gtest))
            for b, name in [("bl", "B-L"), ("ba", "B-A"), ("bn", "B-N")]:
                if b in singles:
                    add_row(name, "-", None, test_eval([1.0], [singles[b]], Ytest, gtest))
            fitted = {}
            for name in dicts:
                rules, Chat_cal = fit_rules(P[name]["cal"], Ycal, gcal)
                fitted[name] = rules
                for rn in ["W-S", "W-E", "W-B", "W-D", "W-Q2", "W-Q5"]:
                    w, info = rules[rn]
                    add_row(rn, name, w, test_eval(w, P[name]["test"], Ytest, gtest),
                            {"g_cal_5": info["g_cal_5"], "risk_cal": info["risk_cal"]})
                # post-hoc optimum on the test matrix (optimistic reference, not a method)
                Ctest, _, _ = cal.residual_matrix(P[name]["test"], Ytest, scale, gtest)
                q = cal.budgeted_qp(Ctest, 5)
                add_row("posthoc-opt", name, q["w"], test_eval(q["w"], P[name]["test"], Ytest, gtest))
                if name == "D-M":
                    # mean single-member risk of each dictionary for eq.(3) context
                    for dn in dicts:
                        Ct, _, _ = cal.residual_matrix(P[dn]["test"], Ytest, scale, gtest)
                        e01_rows.append({"seed": seed, "cond": lab, "method": "mean-member", "dictionary": dn,
                                         "nmse": float(np.mean(np.diag(Ct))), "r2": np.nan, "support_size": 1})
            # --- E02: geometry on D-M (and the other dictionaries for contrast)
            geo = {}
            for name in dicts:
                Ccal, _, _ = cal.residual_matrix(P[name]["cal"], Ycal, scale, gcal)
                Ctest, _, _ = cal.residual_matrix(P[name]["test"], Ytest, scale, gtest)
                wE, wQ = fitted[name]["W-E"][0], fitted[name]["W-Q5"][0]
                wit = cal.two_member_witness(Ccal)
                ent = {"C_cal": Ccal.tolist(), "C_test": Ctest.tolist(),
                       "decomp_equal_test": cal.ambiguity_decomposition(wE, Ctest),
                       "decomp_WQ5_test": cal.ambiguity_decomposition(wQ, Ctest),
                       "witness_cal": {k: v for k, v in wit.items() if k != "all_partners"},
                       "value_curve_test": cal.value_curve(Ctest), "value_curve_cal": cal.value_curve(Ccal)}
                if wit["exists"]:
                    a, b, al = wit["best"], wit["partner"], wit["alpha"]
                    wab = np.zeros(5); wab[a] = 1 - al; wab[b] = al
                    Ra_test = Ctest[a, a]
                    ent["witness_test"] = {"R_a_test": float(Ra_test), "R_mix_test": cal.risk(wab, Ctest),
                                           "realized_gain": float(Ra_test - cal.risk(wab, Ctest)),
                                           "predicted_gain_cal": wit["gain"]}
                geo[name] = ent
            geometry[f"s{seed}|{lab}"] = geo
            # --- E03: m / n_cal slice (D-M, k = 5), conditions N and O1-M only
            if lab in ("N", O1M):
                budgets = M_SLICE_SIM if sim else NCAL_SLICE_SARCOS
                for g in range(N_GROUPS):
                    for mb in budgets:
                        idx = cal_subset(cal_role, mb, g)
                        pc = [p[idx] for p in P["D-M"]["cal"]]
                        gc = gcal[idx] if gcal is not None else None
                        rules, _ = fit_rules(pc, Ycal[idx], gc)
                        for rn in ["W-S", "W-E", "W-B", "W-D", "W-Q5"]:
                            w, info = rules[rn]
                            te = test_eval(w, P["D-M"]["test"], Ytest, gtest)
                            e03_rows.append({"seed": seed, "cond": lab, "group": g, "m": mb, "rule": rn,
                                             "nmse": te["nmse"], "r2": te["r2"], "g_cal_5": info["g_cal_5"],
                                             "risk_cal": info["risk_cal"], "support_size": info["support_size"],
                                             "support": str(info.get("support", "")),
                                             "w": json.dumps([round(float(x), 4) for x in w])})
                            e01_units[(seed, lab, f"E03:{rn}:m{mb}:g{g}", "D-M")] = te["unit_nmse"]
                    # m = 0: dev-selected single (B-F), no target labels
                    w = np.eye(5)[SURROGATE_NAMES.index(R.strong)]
                    te = test_eval(w, P["D-M"]["test"], Ytest, gtest)
                    e03_rows.append({"seed": seed, "cond": lab, "group": g, "m": 0, "rule": "B-F(m=0)",
                                     "nmse": te["nmse"], "r2": te["r2"], "g_cal_5": np.nan, "risk_cal": np.nan,
                                     "support_size": 1, "support": "", "w": json.dumps(w.tolist())})
                # --- E04: k slice on every dictionary at the full budget, per group
                for name in dicts:
                    for g in range(N_GROUPS):
                        idx = cal_subset(cal_role, budgets[-1], g)
                        pc = [p[idx] for p in P[name]["cal"]]
                        gc = gcal[idx] if gcal is not None else None
                        rules, _ = fit_rules(pc, Ycal[idx], gc)
                        for k in range(1, 6):
                            w, info = rules[f"W-Q{k}"]
                            te = test_eval(w, P[name]["test"], Ytest, gtest)
                            e04_rows.append({"seed": seed, "cond": lab, "dictionary": name, "group": g, "k": k,
                                             "nmse": te["nmse"], "r2": te["r2"], "support_size": info["support_size"],
                                             "support": str(info["support"]), "vhat_k": info["vhat_k"],
                                             "kkt": info["kkt"], "w": json.dumps([round(float(x), 4) for x in w])})
                            e01_units[(seed, lab, f"E04:{name}:k{k}:g{g}", name)] = te["unit_nmse"]
                        if name == "D-M" and lab == O1M:
                            weights_out.setdefault(f"s{seed}", {})[f"g{g}"] = {
                                "W-S": rules["W-S"][0].tolist(), "W-Q2": rules["W-Q2"][0].tolist(),
                                "W-Q5": rules["W-Q5"][0].tolist()}
                        if name in ("D-S", "D-T") and lab == O1M:
                            weights_out.setdefault(f"s{seed}", {}).setdefault(f"g{g}", {})[f"{name}:W-Q2"] = rules["W-Q2"][0].tolist()
                            weights_out[f"s{seed}"][f"g{g}"][f"{name}:W-Q5"] = rules["W-Q5"][0].tolist()
                # --- E07 (sim): independent trajectories vs dense sampling at equal frame count
                if sim and lab == O1M:
                    for g in range(N_GROUPS):
                        for m_ind, m_dense in [(1, 10), (2, 10)]:
                            idx_i = cal_subset(cal_role, m_ind, g)
                            idx_d = cal_subset(cal_role, m_dense, g)
                            # subsample dense set to the same frame count: every (m_dense/m_ind)-th frame
                            tr = D["groups"][cal_role][idx_d]
                            keep = np.concatenate([np.where(tr == t)[0][:: m_dense // m_ind] for t in np.unique(tr)])
                            idx_d = idx_d[keep]
                            for tag, idx in [("independent", idx_i), ("dense", idx_d)]:
                                pc = [p[idx] for p in P["D-M"]["cal"]]
                                rules, _ = fit_rules(pc, Ycal[idx], gcal[idx])
                                for rn in ["W-Q5", "W-B", "W-S"]:
                                    w, info = rules[rn]
                                    te = test_eval(w, P["D-M"]["test"], Ytest, gtest)
                                    e07_rows.append({"seed": seed, "group": g, "m_ind": m_ind, "frames": len(idx),
                                                     "sampling": tag, "rule": rn, "nmse": te["nmse"],
                                                     "support_size": info["support_size"]})
            # --- E05 boundary (P1 and O1-H): relative vs absolute, post-hoc optimum, common residual
            if lab in ("P1", f"O1:{frozen['O1_H']}", O1M, "N"):
                Ctest, _, _ = cal.residual_matrix(P["D-M"]["test"], Ytest, scale, gtest)
                wS, wQ = fitted["D-M"]["W-S"][0], fitted["D-M"]["W-Q5"][0]
                tS, tQ = test_eval(wS, P["D-M"]["test"], Ytest, gtest), test_eval(wQ, P["D-M"]["test"], Ytest, gtest)
                # common residual component: projection of e_1 on the orthogonal complement of span{e_j - e_1}
                E = cal.normalized_residuals(P["D-M"]["test"], Ytest, scale).reshape(5, -1)
                diffs = (E[1:] - E[0]).T
                Qm, _ = np.linalg.qr(diffs)
                b = E[0] - Qm @ (Qm.T @ E[0])
                common = float(np.mean(b ** 2) * E.shape[1] / len(Ytest))
                e05_rows.append({"seed": seed, "cond": lab, "nmse_WS": tS["nmse"], "nmse_WQ5": tQ["nmse"],
                                 "rel_gain": 1 - tQ["nmse"] / tS["nmse"], "posthoc_opt": cal.budgeted_qp(Ctest, 5)["value"],
                                 "best_member_test": float(np.min(np.diag(Ctest))), "common_residual_nmse": common,
                                 "r2_WS": tS["r2"], "r2_WQ5": tQ["r2"],
                                 "rmse_joint_WQ5_Nm": json.dumps([round(v, 3) for v in tQ["rmse_joint"]]),
                                 "rmse_joint_WS_Nm": json.dumps([round(v, 3) for v in tS["rmse_joint"]])})
        # theory quantities (sim): B_e^2 from the torque limits, eps_m
        if sim and seed == seeds[0]:
            from sim.panda_env import TORQUE_LIMIT
            Be2 = cal.B_e2_from_limits(TORQUE_LIMIT, D["y_std"], d)
            theory = {"B_e2": Be2, "eps_m": {m: cal.eps_m(Be2, m, 5, 0.05) for m in [1, 2, 5, 10, 100]},
                      "note": "eps_m from eq.(18) with delta = 0.05, M = 5; B_e from clipped predictors and torque limits"}
    # ---------------- summaries with cluster bootstrap over seeds and units
    df = pd.DataFrame(e01_rows); df.to_csv(os.path.join(args.out, "e01_test_table.csv"), index=False)
    if len(df) == 0:
        print("no complete seed; nothing to summarize"); return
    summ = []
    for (lab, method, dn), sub in df.groupby(["cond", "method", "dictionary"]):
        vals, cl_seed, cl_unit = [], [], []
        for s in sub["seed"].unique():
            u = e01_units.get((int(s), lab, method, dn))
            if u is None: continue
            vals.append(np.asarray(u)); cl_seed.append(np.full(len(u), s)); cl_unit.append(np.arange(len(u)))
        if not vals:
            continue
        v = np.concatenate(vals); pt, lo, hi = cluster_bootstrap(v, {"seed": np.concatenate(cl_seed), "unit": np.concatenate(cl_unit)}, n_boot=500)
        summ.append({"cond": lab, "method": method, "dictionary": dn, "nmse_mean": pt, "ci_lo": lo, "ci_hi": hi,
                     "r2_mean": sub["r2"].mean(), "n_seeds": len(vals), "seed_std": sub["nmse"].std(ddof=1) if len(sub) > 1 else np.nan})
    pd.DataFrame(summ).to_csv(os.path.join(args.out, "e01_summary.csv"), index=False)
    pd.DataFrame(e03_rows).to_csv(os.path.join(args.out, "e03_m_slice.csv"), index=False)
    pd.DataFrame(e04_rows).to_csv(os.path.join(args.out, "e04_k_slice.csv"), index=False)
    pd.DataFrame(e05_rows).to_csv(os.path.join(args.out, "e05_boundary.csv"), index=False)
    if e07_rows:
        pd.DataFrame(e07_rows).to_csv(os.path.join(args.out, "e07_density.csv"), index=False)
    with open(os.path.join(args.out, "e02_geometry.json"), "w") as f:
        json.dump(geometry, f, indent=1)
    if theory:
        with open(os.path.join(args.out, "theory_quantities.json"), "w") as f:
            json.dump(theory, f, indent=1)
    if weights_out:
        with open(os.path.join(args.out, "calibration_weights.json"), "w") as f:
            json.dump(weights_out, f)
    # unit-level arrays for paired tests, saved compactly
    np.savez_compressed(os.path.join(args.out, "unit_nmse.npz"),
                        **{"|".join(map(str, k)): np.asarray(v) for k, v in e01_units.items()})
    print("done", len(e01_rows), "E01 rows", len(e03_rows), "E03 rows", len(e04_rows), "E04 rows")


if __name__ == "__main__":
    main()
