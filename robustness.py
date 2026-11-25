"""
Robustness analyses and credit-union scale analysis (companion to model.py).
Run: python3 robustness.py  -> writes robust.json, figC2.png, figF.png, figG.png
All inputs are published figures; derived quantities are computed here.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

SEED, N = 20261123, 200_000
rng = np.random.default_rng(SEED)
D0 = 0.218
W_CLERK = 23.97 / 0.700 / 60          # USD per minute (BLS OEWS May 2025 median; ECEC wage share 70.0%)
W_OFFICER = 36.57 / 0.700 / 60
ST = 6.25                              # minutes saved at the reference case (25-minute task, 25% saved)
out = {"seed": SEED, "N": N}

# ------------------------------------------------------------------
# 1. Second source for detection and checking speed: Irons et al. (2026)
#    q = 0.30 of answers incorrect; overall accuracy (Acc) and acceptance (A) by condition;
#    detection D = (Acc - A + q) / (2q); time per judgement in seconds (correct responses).
# ------------------------------------------------------------------
q = 0.30
IR = {"Full source document": (0.857, 0.715, 45.8), "Top-three passages": (0.831, 0.647, 33.0), "AI-written explanation": (0.808, 0.865, 25.4)}
irons = {}
for k, (acc, A, t) in IR.items():
    D = (acc - A + q) / (2 * q)
    c = (acc - q * D) / (1 - q)
    irons[k] = {"accuracy": acc, "acceptance": A, "seconds": t, "detection": round(D, 3), "correct_accepted": round(c, 3)}
base = irons["AI-written explanation"]
kap_ir = {}
for k in ("Full source document", "Top-three passages"):
    D1, t1 = base["detection"], base["seconds"]
    D2, t2 = irons[k]["detection"], irons[k]["seconds"]
    kap_ir[k] = round(float(-np.log((1 - D2) / (1 - D1)) / ((t2 - t1) / 60)), 2)
out["irons"] = irons
out["kappa_irons"] = kap_ir
KAPPAS = {"Kim et al., low": 0.19, "Kim et al., high": 0.27,
          "Irons et al., full source": kap_ir["Full source document"], "Irons et al., top passages": kap_ir["Top-three passages"]}
DMAX = {"Dell'Acqua et al.": 0.706, "Irons et al., full source": irons["Full source document"]["detection"],
        "Irons et al., top passages": irons["Top-three passages"]["detection"]}
out["dmax_values"] = DMAX

# ------------------------------------------------------------------
# 2. Detection with a ceiling: d(v) = dmax - (dmax - d0) exp(-kappa v)
#    Minimum checking time to reach the high-stakes bar d_inf = 1 - e_h/e.
# ------------------------------------------------------------------
plateau = []
for e in (0.17, 0.33):
    for eh in (0.05, 0.10):
        dinf = 1 - eh / e
        for dn, dm in DMAX.items():
            row = {"e": e, "e_h": eh, "d_inf": round(dinf, 3), "dmax_source": dn, "dmax": dm}
            for kn, k in KAPPAS.items():
                if dm <= dinf or dinf <= D0:
                    v = None if dm <= dinf else 0.0
                else:
                    v = float(-np.log((dm - dinf) / (dm - D0)) / k)
                row[kn] = None if v is None else round(v, 2)
            plateau.append(row)
out["plateau_min_minutes"] = plateau

# ------------------------------------------------------------------
# 3. Regime maps: optimal policy over (kappa, L), exponential and plateau detection
# ------------------------------------------------------------------
def regime(e, eh, dmax=1.0, w=W_CLERK, sT=ST, kmin=-2, kmax=1.2):
    kap = np.logspace(kmin, kmax, 260); L = np.logspace(0, 5, 260)
    K, LL = np.meshgrid(kap, L)
    best = np.full(K.shape, -np.inf); vb = np.zeros(K.shape)
    for v in np.linspace(0, sT, 301):
        d = dmax - (dmax - D0) * np.exp(-K * v)
        enb = w * (sT - v) - (e * (1 - d) - eh) * LL
        m = enb > best; best[m] = enb[m]; vb[m] = v
    return kap, L, np.where(best < 0, 2, np.where(vb < 1e-9, 0, 1))

cmap = ListedColormap(["#8fd19e", "#ffd966", "#e06666"])
legend = [Patch(fc="#8fd19e", label="Use without deliberate checking"), Patch(fc="#ffd966", label="Use with verification"),
          Patch(fc="#e06666", label="Withhold"), Patch(fc="#1f4e79", alpha=0.3, label="κ, Kim et al."), Patch(fc="#6b4c9a", alpha=0.3, label="κ, Irons et al.")]
def draw(fname, dmax, title_extra=""):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey=True)
    shares = {}
    for ax, (e, eh) in zip(axes.ravel(), [(0.17, 0.05), (0.17, 0.155), (0.33, 0.05), (0.33, 0.155)]):
        kap, L, reg = regime(e, eh, dmax)
        ax.pcolormesh(kap, L, reg, cmap=cmap, vmin=0, vmax=2, shading="auto", rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axvspan(0.19, 0.27, color="#1f4e79", alpha=0.3, lw=0)
        ax.axvspan(min(kap_ir.values()), max(kap_ir.values()), color="#6b4c9a", alpha=0.3, lw=0)
        for Lr in (100, 1000, 10000): ax.axhline(Lr, color="#333", lw=0.8, ls=":")
        ax.set_title(f"Assistant error {e:.2f}; unaided error {eh:.3g}{title_extra}", fontsize=10.5)
        shares[f"e={e}|e_h={eh}"] = {n: round(float((reg == i).mean()), 3) for i, n in enumerate(["use_unverified", "verify", "withhold"])}
    for ax in axes[1]: ax.set_xlabel("Verifiability: detection hazard per minute of checking (κ)", fontsize=10)
    for ax in axes[:, 0]: ax.set_ylabel("Consequentiality: cost of an\nundetected error, USD (L)", fontsize=10)
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0.045, 1, 1)); plt.savefig(fname, dpi=300, facecolor="white"); plt.close()
    return shares
out["regime_shares_exponential"] = draw("figC2.png", 1.0)
out["regime_shares_plateau_0807"] = draw("figF.png", DMAX["Irons et al., top passages"], "; ceiling 0.81")

# withhold threshold at high kappa under plateau: is e(1-dmax) > e_h ?
out["plateau_feasible_high_stakes"] = {f"e={e}|e_h={eh}|dmax={dm}": bool(e * (1 - dm) <= eh) for e in (0.17, 0.33) for eh in (0.05, 0.10, 0.155) for dm in DMAX.values()}

# ------------------------------------------------------------------
# 4. Correlated parameters: tools that err more are also harder to catch
#    Gaussian copula between assistant error e and detection d, rho <= 0.
# ------------------------------------------------------------------
from scipy.special import ndtr as ncdf  # noqa: E402
corr = {}
s = rng.uniform(0.13, 0.40, N); T = rng.uniform(20, 30, N)
z1 = rng.standard_normal(N); z2 = rng.standard_normal(N)
for rho in (0.0, -0.5, -0.8):
    za = z1; zb = rho * z1 + np.sqrt(1 - rho ** 2) * z2
    e = 0.17 + 0.16 * ncdf(za)
    for reg, (lo, hi) in (("A", (0.172, 0.318)), ("B", (0.600, 0.706))):
        d = lo + (hi - lo) * ncdf(zb)
        for eh in (0.05, 0.10):
            dr = e * (1 - d) - eh; pos = dr > 0
            Ls = W_CLERK * s[pos] * T[pos] / dr[pos]
            enb100 = W_CLERK * s * T - dr * 100; enb1000 = W_CLERK * s * T - dr * 1000
            corr[f"rho={rho}|{reg}|e_h={eh}"] = {
                "no_threshold": round(float((~pos).mean()), 4),
                "median": None if not pos.any() else round(float(np.median(Ls)), 2),
                "P100": round(float((enb100 > 0).mean()), 4), "P1000": round(float((enb1000 > 0).mean()), 4)}
out["correlated"] = corr

# ------------------------------------------------------------------
# 5. Heavy-tailed error costs: lognormal with the same mean
#    Excess undetected errors per 1,000 instances ~ Poisson(1000 * dr); cost per error lognormal(mean Lbar, sigma)
#    Lbar set at the break-even value, so expected loss equals expected time value saved.
# ------------------------------------------------------------------
e, d, eh = 0.17, 0.60, 0.05
dr = e * (1 - d) - eh
savings = 1000 * W_CLERK * ST
Lbar = W_CLERK * ST / dr
heavy = {"dr": round(dr, 4), "savings_per_1000": round(savings, 2), "Lbar": round(Lbar, 2), "rows": []}
R = 40_000
counts = rng.poisson(1000 * dr, R)
for sig in (0.0, 0.5, 1.0, 1.5, 2.0):
    mu = np.log(Lbar) - sig ** 2 / 2
    tot = np.array([np.exp(rng.normal(mu, sig, c)).sum() if sig > 0 else c * Lbar for c in counts])
    heavy["rows"].append({"sigma": sig, "mean": round(float(tot.mean()), 0), "p50": round(float(np.median(tot)), 0),
                          "p99": round(float(np.percentile(tot, 99)), 0), "p99_over_mean": round(float(np.percentile(tot, 99) / tot.mean()), 2),
                          "P_loss_gt_savings": round(float((tot > savings).mean()), 3), "P_loss_gt_2x_savings": round(float((tot > 2 * savings).mean()), 3)})
out["heavy_tail"] = heavy

# ------------------------------------------------------------------
# 6. Credit-union scale: who bears the fixed cost of measuring the model's inputs
#    NCUA Quarterly Credit Union Data Summary, 2026 Q2. Middle tier by subtraction from system totals.
# ------------------------------------------------------------------
tiers = {"Under USD 100 million": (2438, 73.8, 6.3), "USD 100–500 million": (4214 - 2438 - 748, 2500 - 73.8 - 2186.3, 146.1 - 6.3 - 124.2),
         "Over USD 500 million": (748, 2186.3, 124.2)}
# measurement programme per task class (95% confidence): unaided error near 5% within +/-2 points;
# assistant error near 25% within +/-4 points; detection near 50% within +/-7 points (seeded errors)
z = 1.959964
n_eh = int(np.ceil(z ** 2 * 0.05 * 0.95 / 0.02 ** 2)); n_e = int(np.ceil(z ** 2 * 0.25 * 0.75 / 0.04 ** 2)); n_d = int(np.ceil(z ** 2 * 0.25 / 0.07 ** 2))
REVIEW_MIN, CLASSES = 10, 4
items = n_eh + n_e + n_d
F = items * REVIEW_MIN * W_OFFICER * CLASSES
poolable = n_e / items
scale = {"n_eh": n_eh, "n_e": n_e, "n_d": n_d, "items_per_class": items, "review_minutes": REVIEW_MIN, "classes": CLASSES,
         "reviewer_wage_per_min": round(W_OFFICER, 4), "F_usd": round(F, 0), "poolable_share": round(poolable, 3), "tiers": {}}
tot_cu, tot_mem = 4214, 146.1
for k, (n, a, m) in tiers.items():
    mem_per = m * 1e6 / n
    scale["tiers"][k] = {"credit_unions": n, "share_cu": round(n / tot_cu, 3), "assets_bn": round(a, 1), "members_m": round(m, 1),
                         "share_members": round(m / tot_mem, 3), "members_per_cu": round(mem_per, 0),
                         "cost_per_member": round(F / mem_per, 2),
                         "cost_per_member_pooled20": round(F * (1 - poolable + poolable / 20) / mem_per, 2)}
small, big = scale["tiers"]["Under USD 100 million"], scale["tiers"]["Over USD 500 million"]
scale["ratio_small_to_complex"] = round(small["cost_per_member"] / big["cost_per_member"], 1)
out["scale"] = scale

fig, ax = plt.subplots(figsize=(9.5, 4.4))
names = list(scale["tiers"])
vals = [scale["tiers"][k]["cost_per_member"] for k in names]; pooled = [scale["tiers"][k]["cost_per_member_pooled20"] for k in names]
x = np.arange(3)
b1 = ax.bar(x - 0.18, vals, 0.36, color="#d9822b", label="Each credit union measures alone")
b2 = ax.bar(x + 0.18, pooled, 0.36, color="#1f6fb2", label="Assistant error rate pooled across 20 credit unions")
for b, v in list(zip(b1, vals)) + list(zip(b2, pooled)):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"USD {v:,.2f}", ha="center", fontsize=9)
ax.set_yscale("log"); ax.set_ylim(0.05, 60)
ax.set_xticks(x); ax.set_xticklabels([f"{k}\n({scale['tiers'][k]['credit_unions']:,} credit unions)" for k in names], fontsize=9.5)
ax.set_ylabel("One-off measurement cost per member, USD (log scale)", fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figG.png", dpi=300, facecolor="white"); plt.close()

json.dump(out, open("robust.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("irons", "kappa_irons", "plateau_feasible_high_stakes", "heavy_tail", "scale")}, indent=1))
print(json.dumps(out["correlated"], indent=1))
print(json.dumps(out["regime_shares_exponential"], indent=1), json.dumps(out["regime_shares_plateau_0807"], indent=1))
for r in out["plateau_min_minutes"]: print(r)
