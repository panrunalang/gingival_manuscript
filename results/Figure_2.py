import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== PARAMETERS ====================

# Data from binary logistic regression (Table 4)
variables = [
    'Education level\n(Bachelor & above\nvs Junior college\n& below)',
    'Family annual\nincome (≥300k vs\n<300k CNY)',
    'Age (≥45 vs\n<45 years)',
    'Sex (Female\nvs Male)',
]
or_vals   = [3.21, 2.89, 0.98, 1.12]
ci_low    = [1.12, 1.01, 0.94, 0.45]
ci_high   = [9.21, 8.27, 1.02, 2.79]
p_vals    = [0.030, 0.047, 0.312, 0.806]

# Colors
BG        = '#ffffff'
TEXT_CLR  = '#252525'
LINE_CLR  = '#252525'
REF_CLR   = '#A8D1E1'
SIG_DOT   = '#147EBC'
NS_DOT    = '#A8D1E1'
TEXT_X    = 18            # x position for annotations (data coords, log scale)

FIG_W_MM  = 168
FIG_H_MM  = 78
DPI       = 300

# ==================== SETUP ====================

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.edgecolor'] = LINE_CLR
plt.rcParams['xtick.color'] = TEXT_CLR
plt.rcParams['ytick.color'] = TEXT_CLR

fig_w = FIG_W_MM / 25.4
fig_h = FIG_H_MM / 25.4
fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
ax.set_facecolor(BG)

n = len(variables)
y_positions = np.arange(n)[::-1]

# ==================== DRAW ====================

# Reference line
ax.axvline(x=1, color=REF_CLR, linewidth=1.0, linestyle='--', zorder=1)

# CI lines, dots, and annotation text — all in data coords
for i, (y, or_val, lo, hi, p) in enumerate(zip(y_positions, or_vals, ci_low, ci_high, p_vals)):
    color = SIG_DOT if p < 0.05 else NS_DOT
    # CI line
    ax.plot([lo, hi], [y, y], color=color, linewidth=2.5, solid_capstyle='round', zorder=2)
    # OR dot
    ax.scatter(or_val, y, s=48, color=color, zorder=3, edgecolors='none')
    # Annotation text at fixed x (data coords)
    label = f'{or_val:.2f} ({lo:.2f}–{hi:.2f}), $P$ = {p:.3f}'
    ax.text(TEXT_X, y, label, ha='left', va='center', fontsize=7, color=TEXT_CLR)

# ==================== AXES ====================

ax.set_yticks(y_positions)
ax.set_yticklabels(variables, fontsize=7.5, color=TEXT_CLR)
ax.set_xscale('log')
ax.set_xlim(0.25, 25)   # extend right to fit annotation text
ax.set_xticks([0.5, 1, 2, 4, 8])
ax.set_xticklabels(['0.5', '1', '2', '4', '8'], fontsize=7, color=TEXT_CLR)
ax.set_xlabel('Odds Ratio (95% CI)', fontsize=8, color=TEXT_CLR, labelpad=2)

ax.tick_params(axis='both', length=3, width=0.7, colors=TEXT_CLR)
ax.tick_params(axis='y', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# ==================== SAVE ====================

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'Figure_2.png')
fig.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.04,
            facecolor=BG, edgecolor='none')
plt.close()
print(f'Saved {out}')
