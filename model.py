"""
Replication code: evidence-calibrated decision model for internal GenAI assistants.
All parameters are taken from published sources listed in PARAMS; no primary data are used.
Run: python3 model.py   -> writes results.json, parameters.csv, figA..figE.png
"""
import json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

SEED, N = 20261116, 200_000
rng = np.random.default_rng(SEED)

# ---------------- published parameters ----------------
PARAMS = [
 # symbol, description, low, high, source
 ("s",   "Share of unaided task time saved", 0.13, 0.40, "Brynjolfsson et al. (15% throughput gain = 13% time); Dell'Acqua et al. (25.1%); Noy & Zhang (40%)"),
 ("T",   "Unaided task time, minutes", 20, 30, "Noy & Zhang (20- to 30-minute writing tasks)"),
 ("e",   "Probability that assistant output contains a material error", 0.17, 0.33, "Magesh et al. (retrieval-grounded legal research tools)"),
 ("d_A", "Detection probability, unsupported quick review", 0.172, 0.318, "Kim et al. (accuracy when the LLM answer was incorrect, four display conditions)"),
 ("d_B", "Detection probability, professionals on an analytic task", 0.600, 0.706, "Dell'Acqua et al. (share correct on the outside-frontier task, two AI conditions)"),
 ("e_h", "Unaided human error probability (scenario values)", 0.05, 0.155, "Upper value: Dell'Acqua et al. control group (84.5% correct); lower values are scenarios"),
 ("kappa", "Detection hazard per minute of checking", 0.19, 0.27, "Derived from Kim et al. (change in accuracy and time when sources were shown)"),
 ("d0",  "Detection with no deliberate checking", 0.218, 0.218, "Kim et al. (incorrect answer, no explanation, no sources)"),
 ("w_teller", "Loaded labour cost per minute, tellers (USD)", None, None, "BLS OEWS May 2025 median USD 20.63/h; BLS ECEC June 2026 wage share 70.0%"),
 ("w_clerk",  "Loaded labour cost per minute, loan interviewers and clerks (USD)", None, None, "BLS OEWS May 2025 median USD 23.97/h; ECEC as above"),
 ("w_officer","Loaded labour cost per minute, loan officers (USD)", None, None, "BLS OEWS May 2025 median USD 36.57/h; ECEC as above"),
]
WAGE = {"Tellers": 20.63, "Loan interviewers and clerks": 23.97, "Loan officers": 36.57}
LOAD = 1 / 0.700
W = {k: v * LOAD / 60 for k, v in WAGE.items()}           # USD per minute
E_H = [0.05, 0.10, 0.155]
REF_L = {"EFTA minimum": 100, "EFTA maximum": 1000, "ECOA punitive cap": 10000}

# kappa implied by Kim et al.: 1-d(v) = (1-d0) exp(-kappa v)
k1 = -np.log((1 - 0.318) / (1 - 0.218)) / (1.39 - 0.89)   # no explanation: neither -> sources
k2 = -np.log((1 - 0.231) / (1 - 0.172)) / (1.44 - 1.05)   # explanation: without -> with sources
KAPPA = (min(k1, k2), max(k1, k2))
D0 = 0.218

res = {"seed": SEED, "N": N, "w_per_min": W, "kappa_implied": [round(k1, 3), round(k2, 3)]}

# ---------------- Result 2: asymptotic detection requirement ----------------
res["d_inf"] = {f"e={e}": {f"e_h={eh}": round(1 - eh / e, 3) for eh in E_H} for e in (0.17, 0.33)}

# ---------------- Monte Carlo: break-even loss ----------------
s = rng.uniform(0.13, 0.40, N); T = rng.uniform(20, 30, N); e = rng.uniform(0.17, 0.33, N)
dA = rng.uniform(0.172, 0.318, N); dB = rng.uniform(0.600, 0.706, N)
mc = {}
box = {}
for reg, d in (("A", dA), ("B", dB)):
    for eh in E_H:
        dr = e * (1 - d) - eh
        pos = dr > 0
        row = {"share_no_worse_than_human": round(float((~pos).mean()), 4)}
        for occ, w in W.items():
            Ls = w * s[pos] * T[pos] / dr[pos]
            q = np.percentile(Ls, [5, 50, 95]) if pos.any() else [np.nan] * 3
            pr = {}
            for name, L in REF_L.items():
                enb = w * s * T - dr * L
                pr[name] = round(float((enb > 0).mean()), 4)
            f2 = lambda x: None if np.isnan(x) else round(float(x), 2)
            row[occ] = {"p5": f2(q[0]), "median": f2(q[1]), "p95": f2(q[2]), "P_ENB_pos": pr}
            if occ == "Loan interviewers and clerks":
                box[(reg, eh)] = Ls
        mc[f"{reg}|e_h={eh}"] = row
res["monte_carlo"] = mc

# sensitivity (Spearman) for regime A, e_h = 0.05, clerk wage
def rank(x): return np.argsort(np.argsort(x)).astype(float)
def spearman(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
dr = e * (1 - dA) - 0.05; pos = dr > 0
Lstar = W["Loan interviewers and clerks"] * s[pos] * T[pos] / dr[pos]
sens = {"Time saved share (s)": spearman(s[pos], Lstar), "Unaided task time (T)": spearman(T[pos], Lstar),
        "Assistant error rate (e)": spearman(e[pos], Lstar), "Detection probability (d)": spearman(dA[pos], Lstar)}
res["sensitivity_spearman_A_eh0.05"] = {k: round(v, 3) for k, v in sens.items()}

# ---------------- Result 4: residual-error floor ----------------
floor = []
for e0 in (0.17, 0.33):
    for sT in (3.25, 6.25, 10.0):
        for kap in KAPPA:
            r = e0 * (1 - D0) * np.exp(-kap * sT)
            floor.append({"e": e0, "sT": sT, "kappa": round(float(kap), 3), "r_floor": round(float(r), 4), "d_max": round(float(1 - r / e0), 3)})
res["residual_floor"] = floor
res["kappa_critical"] = {f"e={e0}|e_h={eh}": round(float(np.log(e0 * (1 - D0) / eh) / 6.25), 3) for e0 in (0.17, 0.33) for eh in E_H}

# ---------------- Result 3: cost of miscalibration ----------------
wk = W["Loan interviewers and clerks"]; kmid = float(np.mean(KAPPA))
res["miscalibration"] = {"w_over_kappa_usd": round(wk / kmid, 2),
    "excess_cost": {str(m): {"units": round(float(np.log(m) + 1 / m - 1), 3), "usd": round(float((wk / kmid) * (np.log(m) + 1 / m - 1)), 2)} for m in (0.25, 0.5, 1.0, 2.0, 4.0)}}

# ---------------- worked reference case ----------------
w = wk; sT = 6.25
ref = {}
for e0 in (0.17, 0.33):
    for eh in E_H:
        for lab, d in (("d=0.218", 0.218), ("d=0.318", 0.318), ("d=0.60", 0.60), ("d=0.706", 0.706)):
            drr = e0 * (1 - d) - eh
            ref[f"e={e0}|e_h={eh}|{lab}"] = None if drr <= 0 else round(w * sT / drr, 2)
res["reference_case_Lstar"] = ref
res["reference_saving_usd"] = round(w * sT, 2)

# ---------------- regime map ----------------
def regime_map(e0, eh, sT=6.25, w=wk):
    kap = np.logspace(-2, 0, 220); L = np.logspace(0, 4.5, 240)
    K, LL = np.meshgrid(kap, L)
    vgrid = np.linspace(0, sT, 251)
    best = np.full(K.shape, -np.inf); vbest = np.zeros(K.shape)
    for v in vgrid:
        enb = w * (sT - v) - (e0 * (1 - D0) * np.exp(-K * v) - eh) * LL
        m = enb > best; best[m] = enb[m]; vbest[m] = v
    reg = np.where(best < 0, 2, np.where(vbest < 1e-9, 0, 1))
    return kap, L, reg
shares = {}
fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True, sharey=True)
cmap = ListedColormap(["#8fd19e", "#ffd966", "#e06666"])
for ax, (e0, eh) in zip(axes.ravel(), [(0.17, 0.05), (0.17, 0.155), (0.33, 0.05), (0.33, 0.155)]):
    kap, L, reg = regime_map(e0, eh)
    ax.pcolormesh(kap, L, reg, cmap=cmap, vmin=0, vmax=2, shading="auto", rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvspan(KAPPA[0], KAPPA[1], color="#1f4e79", alpha=0.25, lw=0)
    for name, Lr in REF_L.items(): ax.axhline(Lr, color="#333", lw=0.8, ls=":")
    ax.set_title(f"Assistant error rate {e0:.2f}; unaided human error rate {eh:.3g}", fontsize=10.5)
    shares[f"e={e0}|e_h={eh}"] = {n: round(float((reg == i).mean()), 3) for i, n in enumerate(["use_unverified", "verify", "withhold"])}
for ax in axes[1]: ax.set_xlabel("Verifiability: detection hazard per minute of checking (κ)", fontsize=10)
for ax in axes[:, 0]: ax.set_ylabel("Consequentiality: cost of an\nundetected error, USD (L)", fontsize=10)
axes[0, 0].text(1.05, 110, "USD 100", fontsize=7.5, ha="right"); axes[0, 0].text(1.05, 1100, "USD 1,000", fontsize=7.5, ha="right"); axes[0, 0].text(1.05, 11000, "USD 10,000", fontsize=7.5, ha="right")
fig.legend(handles=[Patch(fc="#8fd19e", label="Use without deliberate checking"), Patch(fc="#ffd966", label="Use with verification"),
                    Patch(fc="#e06666", label="Withhold"), Patch(fc="#1f4e79", alpha=0.25, label="κ implied by Kim et al.")],
           loc="lower center", ncol=4, fontsize=9.5, frameon=False)
plt.tight_layout(rect=(0, 0.045, 1, 1)); plt.savefig("figC.png", dpi=300, facecolor="white"); plt.close()
res["regime_area_shares"] = shares

# ---------------- Figure A: break-even loss vs detection ----------------
fig, ax = plt.subplots(figsize=(10, 5.6))
dgrid = np.linspace(0, 0.99, 500)
cols = {0.17: "#1f6fb2", 0.33: "#d9822b"}; lss = {0.05: "-", 0.155: "--"}
for e0 in (0.17, 0.33):
    for eh in (0.05, 0.155):
        drr = e0 * (1 - dgrid) - eh
        Ls = np.where(drr > 0, wk * 6.25 / np.where(drr > 0, drr, 1), np.nan)
        ax.plot(dgrid, Ls, color=cols[e0], ls=lss[eh], lw=2.2, label=f"e = {e0:.2f}, unaided error = {eh:.3g}")
ax.axvspan(0.172, 0.318, color="#6b4c9a", alpha=0.15); ax.axvspan(0.600, 0.706, color="#2e8b57", alpha=0.15)
ax.text(0.245, 4.2e4, "Unsupported quick\nreview (Kim et al.)", ha="center", fontsize=8.5, color="#4b2f7a")
ax.text(0.653, 4.2e4, "Professionals\n(Dell'Acqua et al.)", ha="center", fontsize=8.5, color="#1e6b40")
for name, Lr in REF_L.items():
    ax.axhline(Lr, color="#555", lw=0.9, ls=":"); ax.text(0.995, Lr * 1.12, f"{name}: USD {Lr:,}", ha="right", fontsize=8, color="#333")
ax.set_yscale("log"); ax.set_ylim(10, 1e5); ax.set_xlim(0, 1)
ax.set_xlabel("Probability that review detects a material error (d)", fontsize=10.5)
ax.set_ylabel("Break-even cost of an undetected error, USD (L*)", fontsize=10.5)
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figA.png", dpi=300, facecolor="white"); plt.close()

# ---------------- Figure B: Monte Carlo distributions ----------------
fig, ax = plt.subplots(figsize=(10, 5.2))
keys = [("A", 0.05), ("A", 0.10), ("A", 0.155), ("B", 0.05), ("B", 0.10)]
data = [np.log10(box[k]) for k in keys]
bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55)
for patch, c in zip(bp["boxes"], ["#6b4c9a", "#8f74b8", "#b9a6d6", "#2e8b57", "#7fc29b"]): patch.set_facecolor(c)
for med in bp["medians"]: med.set_color("black")
ax.set_xticks(range(1, 6)); ax.set_xticklabels([f"Regime {r}\nunaided error {h:.3g}" for r, h in keys], fontsize=9)
ax.set_yticks([1, 2, 3, 4]); ax.set_yticklabels(["10", "100", "1,000", "10,000"])
for Lr in REF_L.values(): ax.axhline(np.log10(Lr), color="#555", lw=0.9, ls=":")
ax.set_ylabel("Break-even cost of an undetected error, USD (log scale)", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figB.png", dpi=300, facecolor="white"); plt.close()

# ---------------- Figure D: sensitivity ----------------
fig, ax = plt.subplots(figsize=(9, 3.6))
items = sorted(sens.items(), key=lambda kv: abs(kv[1]))
ax.barh([k for k, _ in items], [v for _, v in items], color=["#d9822b" if v < 0 else "#1f6fb2" for _, v in items])
for i, (k, v) in enumerate(items): ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=10)
ax.axvline(0, color="black", lw=0.8); ax.set_xlim(-1, 1); ax.set_xlabel("Spearman rank correlation with break-even loss", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig("figD.png", dpi=300, facecolor="white"); plt.close()

# ---------------- Figure E: residual error floor ----------------
fig, ax = plt.subplots(figsize=(10, 5.0))
st = np.linspace(0, 12, 300)
for e0, c in ((0.17, "#1f6fb2"), (0.33, "#d9822b")):
    lo = e0 * (1 - D0) * np.exp(-KAPPA[1] * st); hi = e0 * (1 - D0) * np.exp(-KAPPA[0] * st)
    ax.fill_between(st, lo * 100, hi * 100, color=c, alpha=0.35, label=f"Assistant error rate {e0:.2f} (band: κ from {KAPPA[0]:.2f} to {KAPPA[1]:.2f})")
for eh, ls in ((0.05, "--"), (0.155, ":")):
    ax.axhline(eh * 100, color="#333", ls=ls, lw=1.2); ax.text(11.9, eh * 100 + 0.5, f"Unaided human error {eh*100:.3g}%", ha="right", fontsize=8.5)
for x, lab in ((3.25, "13% saved"), (6.25, "25% saved"), (10, "40% saved")):
    ax.axvline(x, color="#999", lw=0.8); ax.text(x + 0.1, 26.3, lab, fontsize=8.5, color="#555")
ax.set_xlabel("Minutes saved on a 25-minute task, all reinvested in checking (sT)", fontsize=10.5)
ax.set_ylabel("Lowest attainable residual error rate (%)", fontsize=10.5); ax.set_ylim(0, 28); ax.set_xlim(0, 12)
ax.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.6, 0.93)); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figE.png", dpi=300, facecolor="white"); plt.close()

json.dump(res, open("results.json", "w"), indent=1)
with open("parameters.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["symbol", "description", "low", "high", "source"])
    for p in PARAMS: wr.writerow(p)
    for k, v in W.items(): wr.writerow([f"w[{k}]", "computed loaded cost per minute (USD)", round(v, 4), round(v, 4), "OEWS median / 0.700 / 60"])
print("done")
