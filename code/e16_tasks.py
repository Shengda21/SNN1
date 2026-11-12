"""Write the priority-ordered E16-E18 task list (JSON lines) read by e16_train.py.

python e16_tasks.py --out ../logs/e16_tasks.jsonl [--stages e16s0,e16s1,...] [--extra "w16_tri_h1:3,4"]

Stages, in queue order:
  e16s0, e16s1         peak-normalized width sweep, 4 families x 7 HWHM, both platforms
  e16d                 detach-reset variant (forward unchanged, reset spk detached in the
                       backward): tri and atan at HWHM {0.0625, 0.25, 1, 4, 16} plus STE;
                       seeds 0-2 (inserted with top priority after seeds 0-1 had finished)
  e16s2                width sweep, seed 2
  e16c                 control: original parametrization (atan peak a/2, sigmoid peak k/4),
                       same HWHM grid, seed 0 (tri / sre coincide with E16 and are not repeated)
  e17                  triangular layer-wise placement: wide HWHM H in {1, 4} only on layer 1
                       (L1) or only on layers 2-3 (L23), the other layers at 0.25; seeds 0-2
  e16x                 ultra-wide peak-normalized tri and atan, HWHM {8, 16, 32, 64}, seed 0
                       (does the sweep approach STE continuously?)
  e18p                 E18 priority subset, seed 0: membrane sigma {0.5, 2}, augmentation
                       {0.5, 1.4}, the three bounded-activation MLPs (members keep this position)
  e18mem               strong surrogate (frozen width), training-only first-layer membrane
                       noise sigma in {0.25, 0.5, 1, 2}; seeds 0-2
  e18aug               strong surrogate, B-A input augmentation with ceiling
                       {0.25, 0.5, 1, 1.4} x aug_sigma; seeds 0-2
  e18mlp               B-N MLP with tanh / clipped ReLU / binary-STE activation; seeds 0-2
  e18esc (optional)    stochastic escape-noise forward, sigmoid escape function with
                       HWHM 0.25 and 1 (beta = 7.05, 1.76); seeds 0-2
Within a stage the sim member precedes the sarcos member of the same configuration
(sim runs are longer; this balances the shared queue).
--extra adds seeds to named E16 configurations (e.g. after the main agent picks the
configurations that enter the main conclusions): "w16_tri_h1:3,4;w16_sig_h0.25:3,4".
"""
import argparse
import json
import os

from e16_surrogates import FAMILIES, HWHM_GRID, hwhm_str, param_from_hwhm, hwhm_from_param

HERE = os.path.dirname(os.path.abspath(__file__))
PLATFORMS = ["sim", "sarcos"]
PN_SURR_NAME = {"tri": "triangular", "sre": "sre", "sig": "sigmoid_pn", "atan": "atan_pn"}
ORIG_SURR_NAME = {"sig": "sigmoid", "atan": "atan"}
FAM_OF_ORIG = {"triangular": "tri", "sigmoid": "sig", "atan": "atan", "sre": "sre"}
# queue order (coordinator, 16:15): w16d -> ultra-wide -> E17 -> rest of seed 2 and control ->
# E18 priority subset (seed 0: membrane sigma 0.5 / 2, augmentation 0.5 / 1.4, three MLPs) -> rest of E18
STAGES = ["e16s0", "e16s1", "e16d", "e16x", "e17", "e16s2", "e16c", "e18p", "e18mem", "e18aug", "e18mlp", "e18esc"]


def frozen(p):
    return json.load(open(os.path.join(HERE, "..", "runs25", p, "frozen.json")))


def w16(plat, fam, h, seed, block="w16", detach=False):
    pn = block in ("w16", "w16d")
    peak = 1.0 if pn else {"sig": param_from_hwhm("sig", h) / 4.0, "atan": param_from_hwhm("atan", h) / 2.0}[fam]
    return {"platform": plat, "member_id": f"{block}_{fam}_h{hwhm_str(h)}_s{seed}", "block": block,
            "experiment": "E16", "kind": "snn", "seed": seed, "family": fam, "hwhm": float(h),
            "peak_normalized": pn, "peak": peak,
            "surrogate": PN_SURR_NAME[fam] if pn else ORIG_SURR_NAME[fam], "width": param_from_hwhm(fam, h),
            "layers": [[fam, float(h), pn]] * 3, "hwhm_layers": [float(h)] * 3,
            "membrane_noise": 0.0, "aug_scale": 0.0, "activation": "spike", "detach_reset": detach}


def w16d_ste(plat, seed):
    return {"platform": plat, "member_id": f"w16d_ste_s{seed}", "block": "w16d", "experiment": "E16",
            "kind": "snn", "seed": seed, "family": "ste", "hwhm": None, "peak_normalized": True, "peak": 1.0,
            "surrogate": "ste", "width": None, "layers": [["ste", None, True]] * 3, "hwhm_layers": None,
            "membrane_noise": 0.0, "aug_scale": 0.0, "activation": "spike", "detach_reset": True}


def w17(plat, H, placement, seed, narrow=0.25):
    hl = [H, narrow, narrow] if placement == "L1" else [narrow, H, H]
    return {"platform": plat, "member_id": f"w17_tri_{placement}h{hwhm_str(H)}_s{seed}", "block": "w17",
            "experiment": "E17", "kind": "snn", "seed": seed, "family": "tri", "hwhm": float(H),
            "hwhm_narrow": narrow, "placement": placement, "peak_normalized": True, "peak": 1.0,
            "surrogate": "triangular_layerwise", "width": None,
            "layers": [["tri", float(x), True] for x in hl], "hwhm_layers": [float(x) for x in hl],
            "membrane_noise": 0.0, "aug_scale": 0.0, "activation": "spike"}


def strong_base(plat):
    fr = frozen(plat)
    s = fr["strong_surrogate"]; w = fr["widths"][s]
    fam = FAM_OF_ORIG[s]
    h = hwhm_from_param(fam, w)
    peak = {"sig": w / 4.0, "atan": w / 2.0, "tri": 1.0, "sre": 1.0}[fam]
    return {"family": fam, "hwhm": h, "peak_normalized": fam in ("tri", "sre"), "peak": peak,
            "surrogate": s, "width": w, "layers": [["orig", s, w]] * 3, "hwhm_layers": [h] * 3}


def w18mem(plat, sigma, seed):
    t = {"platform": plat, "member_id": f"w18_mem{hwhm_str(sigma)}_s{seed}", "block": "w18mem",
         "experiment": "E18", "kind": "snn", "seed": seed, "membrane_noise": float(sigma),
         "membrane_noise_mode": "layer1_additive_current_train_only", "aug_scale": 0.0, "activation": "spike"}
    t.update(strong_base(plat))
    return t


def w18aug(plat, scale, seed):
    t = {"platform": plat, "member_id": f"w18_aug{hwhm_str(scale)}_s{seed}", "block": "w18aug",
         "experiment": "E18", "kind": "snn", "seed": seed, "membrane_noise": 0.0,
         "aug_scale": float(scale), "activation": "spike"}
    t.update(strong_base(plat))
    return t


def w18mlp(plat, act, seed):
    return {"platform": plat, "member_id": f"w18_mlp_{act}_s{seed}", "block": "w18mlp",
            "experiment": "E18", "kind": "mlp", "seed": seed, "activation": act, "family": None,
            "hwhm": None, "peak_normalized": None, "membrane_noise": 0.0, "aug_scale": 0.0}


def w18esc(plat, h, seed):
    beta = param_from_hwhm("sig", h)
    return {"platform": plat, "member_id": f"w18_esc_h{hwhm_str(h)}_s{seed}", "block": "w18esc",
            "experiment": "E18", "kind": "snn", "seed": seed, "family": "esc", "hwhm": float(h),
            "peak_normalized": False, "peak": beta / 4.0, "surrogate": "escape_sigmoid", "width": beta,
            "layers": [["esc", float(h)]] * 3, "hwhm_layers": [float(h)] * 3, "stochastic_forward": True,
            "membrane_noise": 0.0, "aug_scale": 0.0, "activation": "spike"}


def stage_tasks(st):
    T = []
    if st.startswith("e16s"):
        sd = int(st[4:])
        for fam in FAMILIES:
            for h in HWHM_GRID:
                for p in PLATFORMS:
                    T.append(w16(p, fam, h, sd))
    elif st == "e16d":
        for sd in (0, 1, 2):
            for fam in ["tri", "atan"]:
                for h in [0.0625, 0.25, 1.0, 4.0, 16.0]:
                    for p in PLATFORMS:
                        T.append(w16(p, fam, h, sd, block="w16d", detach=True))
            for p in PLATFORMS:
                T.append(w16d_ste(p, sd))
    elif st == "e16x":
        for fam in ["tri", "atan"]:
            for h in [8.0, 16.0, 32.0, 64.0]:
                for p in PLATFORMS:
                    T.append(w16(p, fam, h, 0))
    elif st == "e16c":
        for fam in ["sig", "atan"]:
            for h in HWHM_GRID:
                for p in PLATFORMS:
                    T.append(w16(p, fam, h, 0, block="w16c"))
    elif st == "e17":
        for sd in (0, 1, 2):
            for H in (1.0, 4.0):
                for pl in ("L1", "L23"):
                    for p in PLATFORMS:
                        T.append(w17(p, H, pl, sd))
    elif st == "e18p":
        for p in PLATFORMS:
            T += [w18mem(p, 0.5, 0), w18mem(p, 2.0, 0), w18aug(p, 0.5, 0), w18aug(p, 1.4, 0)]
            T += [w18mlp(p, a, 0) for a in ("tanh", "crelu", "bin")]
    elif st == "e18mem":
        for sd in (0, 1, 2):
            for s in (0.25, 0.5, 1.0, 2.0):
                for p in PLATFORMS:
                    T.append(w18mem(p, s, sd))
    elif st == "e18aug":
        for sd in (0, 1, 2):
            for s in (0.25, 0.5, 1.0, 1.4):
                for p in PLATFORMS:
                    T.append(w18aug(p, s, sd))
    elif st == "e18mlp":
        for sd in (0, 1, 2):
            for a in ("tanh", "crelu", "bin"):
                for p in PLATFORMS:
                    T.append(w18mlp(p, a, sd))
    elif st == "e18esc":
        for sd in (0, 1, 2):
            for h in (0.25, 1.0):
                for p in PLATFORMS:
                    T.append(w18esc(p, h, sd))
    else:
        raise ValueError(st)
    for t in T:
        t["stage"] = st
    return T


def extra_tasks(spec):
    T = []
    for item in spec.split(";"):
        if not item.strip():
            continue
        cfg, seeds = item.split(":")
        block, fam, hh = cfg.split("_")
        h = float(hh[1:])
        for sd in [int(x) for x in seeds.split(",")]:
            for p in PLATFORMS:
                t = w16(p, fam, h, sd, block=block); t["stage"] = "extra"
                T.append(t)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--extra", default="")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    T = []
    for st in args.stages.split(","):
        if st:
            T += stage_tasks(st)
    T += extra_tasks(args.extra)
    # a member listed by several stages keeps its first (highest-priority) position
    seen, U = set(), []
    for t in T:
        k = (t["platform"], t["member_id"])
        if k not in seen:
            seen.add(k); U.append(t)
    T = U
    with open(args.out, "a" if args.append else "w") as f:
        for t in T:
            f.write(json.dumps(t) + "\n")
    from collections import Counter
    print(len(T), "tasks", dict(Counter(t["stage"] for t in T)))


if __name__ == "__main__":
    main()
