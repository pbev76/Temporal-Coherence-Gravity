import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

OUT = Path("grb_output")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(OUT / "grb_final.csv")

# Clean
df = df[df["z"] > 0.01].copy()
df["T90_corr"] = df["T90"] / (1 + df["z"])
df = df[df["T90_corr"] > 0]

print(f"Analyzing {len(df)} GRBs")

# Binning by redshift (equal count or log bins)
df['z_bin'] = pd.qcut(df['z'], q=12, duplicates='drop')

stats = df.groupby('z_bin', observed=True).agg(
    n=('T90_corr', 'size'),
    mean_z=('z', 'mean'),
    median_z=('z', 'median'),
    std_T90=('T90_corr', 'std'),
    var_T90=('T90_corr', 'var'),
    mean_T90=('T90_corr', 'mean')
).reset_index()

stats = stats[stats['n'] >= 6]  # reasonable minimum per bin

# Correlation between redshift and jitter
corr_pearson = pearsonr(stats['mean_z'], stats['std_T90'])
corr_spearman = spearmanr(stats['mean_z'], stats['std_T90'])

print("\n=== Jitter vs Redshift (Universe Age Proxy) ===")
print(f"Pearson  correlation: {corr_pearson}")
print(f"Spearman correlation: {corr_spearman}")

# Plots
fig, axs = plt.subplots(2, 2, figsize=(14, 11))

# 1. Scatter of all points
axs[0,0].scatter(df['z'], df['T90_corr'], alpha=0.65, s=16, label='Individual GRBs')
axs[0,0].set_xscale('log')
axs[0,0].set_yscale('log')
axs[0,0].set_xlabel('Redshift z (proxy for lookback time)')
axs[0,0].set_ylabel('T90 / (1+z)  [seconds]')
axs[0,0].set_title('Corrected Duration vs Redshift')
axs[0,0].grid(True, alpha=0.3)
axs[0,0].legend()

# 2. Binned Jitter (main result)
axs[0,1].errorbar(stats['mean_z'], stats['std_T90'], 
                  yerr=stats['std_T90']/np.sqrt(stats['n']), fmt='o-', 
                  capsize=5, color='red', linewidth=2.5, markersize=7)
axs[0,1].set_xscale('log')
axs[0,1].set_xlabel('Mean Redshift in Bin')
axs[0,1].set_ylabel('Standard Deviation of T90_corr')
axs[0,1].set_title('Jitter (Spread) vs Redshift\n(Higher = more variation in durations)')
axs[0,1].grid(True, alpha=0.3)

# 3. Variance on log-log
axs[1,0].plot(stats['mean_z'], stats['var_T90'], 's-', color='darkred', linewidth=2)
axs[1,0].set_xscale('log')
axs[1,0].set_yscale('log')
axs[1,0].set_xlabel('Redshift z')
axs[1,0].set_ylabel('Variance of T90_corr')
axs[1,0].set_title('Variance vs Redshift (log-log)')
axs[1,0].grid(True, alpha=0.3)

# 4. Number of events per bin
axs[1,1].bar(range(len(stats)), stats['n'], alpha=0.7)
axs[1,1].set_xlabel('Redshift Bin')
axs[1,1].set_ylabel('Number of GRBs')
axs[1,1].set_title('Sample Size per Bin')
axs[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "causality_jitter_vs_redshift.png", dpi=300, bbox_inches="tight")

print(f"\n✅ Main plot saved: grb_output/causality_jitter_vs_redshift.png")
print("Look especially at the top-right panel (Jitter vs Redshift).")