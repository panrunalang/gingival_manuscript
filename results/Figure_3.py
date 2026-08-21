import os
import csv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

# ==================== PARAMETERS ====================

BG       = '#ffffff'
TEXT_CLR = '#252525'
C3_LIGHT = '#F3CCDB'
C3_MID   = '#A8D1E1'
C3_DARK  = '#147EBC'
C2_YES   = '#147EBC'
C2_NO    = '#E5E5F3'

FIG_W_MM    = 180
FIG_H_MM    = 110
DPI         = 300
TARGET_W_PX = 2126

LABELS = ['A', 'B', 'C']

# ==================== DATA LOADING ====================

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'survey.csv')
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)
    anxiety_raw, postop_raw, psych_raw, fsurg_raw = [], [], [], []
    for row in reader:
        if len(row) >= 26:
            anxiety_raw.append(row[16])
            postop_raw.append(row[20])
            psych_raw.append(row[23])
            fsurg_raw.append(row[25])

# ==================== VARIABLE RECODING ====================

anx_map = {
    'A.完全没有': 'No anxiety',
    'B.轻度焦虑：情绪紧张不安，莫名担心、慌乱': 'Mild\nanxiety',
    'C. 中度焦虑：持续的紧张难安，莫名担心，心慌心悸': 'Moderate\nanxiety',
}
post_map = {
    'A.无特殊不适': 'No\ndiscomfort',
    'B.轻微症状：少量出血、疼痛、肿胀': 'Mild\nsymptoms',
    'C.明显不适：明显疼痛、肿胀、需口服止疼药才能缓解': 'Marked\ndiscomfort',
}
psych_map = {'A:是': 'Yes', 'B:否': 'No'}
fsurg_map = {
    'A.完全不影响': 'No\ninfluence',
    'B.轻微影响，若其他部位有必要，还是会积极接受手术方案': 'Mild\ninfluence',
    'C.比较影响，即使其他部位有必要，还是需要考虑很久才能接受': 'Moderate\ninfluence',
}

anxiety      = [anx_map[v] for v in anxiety_raw]
postop       = [post_map[v] for v in postop_raw]
psych_burden = [psych_map[v] for v in psych_raw]
future_surg  = [fsurg_map[v] for v in fsurg_raw]

# ==================== ORDINAL SCORES & STATISTICS ====================

anx_score = []
for v in anxiety_raw:
    if '完全没有' in v:      anx_score.append(0)
    elif '轻度' in v:        anx_score.append(1)
    elif '中度' in v:        anx_score.append(2)

post_score = []
for v in postop_raw:
    if '无特殊不适' in v:    post_score.append(0)
    elif '轻微' in v:        post_score.append(1)
    elif '明显' in v:        post_score.append(2)

psych_score = [1 if '是' in v else 0 for v in psych_raw]

fs_score = []
for v in fsurg_raw:
    if '完全不影响' in v:    fs_score.append(0)
    elif '轻微' in v:        fs_score.append(1)
    elif '比较' in v:        fs_score.append(2)

tau_a, p_a = kendalltau(anx_score, post_score)
tau_b, p_b = kendalltau(anx_score, psych_score)
tau_c, p_c = kendalltau(post_score, fs_score)

# ==================== CROSS-TAB DATA ====================

anx_levels   = ['No anxiety', 'Mild\nanxiety', 'Moderate\nanxiety']
post_levels  = ['No\ndiscomfort', 'Mild\nsymptoms', 'Marked\ndiscomfort']
psych_levels = ['Yes', 'No']
fs_levels    = ['No\ninfluence', 'Mild\ninfluence', 'Moderate\ninfluence']

post_short   = ['None', 'Mild', 'Marked']

# Panel A
tab_a = Counter(zip(anxiety, postop))
mat_a = np.array([[tab_a.get((ax, po), 0) for po in post_levels] for ax in anx_levels])
mat_a_pct = mat_a / mat_a.sum(axis=1, keepdims=True) * 100

# Panel B
tab_b = Counter(zip(anxiety, psych_burden))
mat_b = np.array([[tab_b.get((ax, py), 0) for py in psych_levels] for ax in anx_levels])
mat_b_pct = mat_b / mat_b.sum(axis=1, keepdims=True) * 100

# Panel C
tab_c = Counter(zip(postop, future_surg))
mat_c = np.array([[tab_c.get((po, fs), 0) for fs in fs_levels] for po in post_levels])
mat_c_pct = mat_c / mat_c.sum(axis=1, keepdims=True) * 100

# ==================== FIGURE SETUP ====================

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8.5

fig_w = TARGET_W_PX / DPI
fig_h = FIG_H_MM / 25.4

fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), facecolor=BG, dpi=DPI)
plt.subplots_adjust(left=0.08, right=0.96, bottom=0.30, top=0.88, wspace=0.32)

panels = [
    {   # A
        'ax': axes[0],
        'mat_pct': mat_a_pct, 'mat_count': mat_a,
        'row_labels': anx_levels,
        'col_labels': ['No discomfort', 'Mild symptoms', 'Marked discomfort'],
        'colors': [C3_LIGHT, C3_MID, C3_DARK],
        'legend_title': 'Postoperative reaction',
        'ncol_legend': 1,
        'p': p_a,
    },
    {   # B
        'ax': axes[1],
        'mat_pct': mat_b_pct, 'mat_count': mat_b,
        'row_labels': anx_levels,
        'col_labels': ['Yes', 'No'],
        'colors': [C2_YES, C2_NO],
        'legend_title': 'Psychological burden',
        'ncol_legend': 2,
        'p': p_b,
    },
    {   # C
        'ax': axes[2],
        'mat_pct': mat_c_pct, 'mat_count': mat_c,
        'row_labels': post_short,
        'col_labels': ['No influence', 'Mild influence', 'Moderate influence'],
        'colors': [C3_LIGHT, C3_MID, C3_DARK],
        'legend_title': 'Future surgery willingness',
        'ncol_legend': 1,
        'p': p_c,
    },
]

# ==================== DRAW EACH PANEL ====================

bar_w = 0.55

for i, pn in enumerate(panels):
    ax = pn['ax']
    ax.set_facecolor(BG)

    n_rows = len(pn['row_labels'])
    n_cols = len(pn['col_labels'])
    x = np.arange(n_rows)

    # --- Stacked bars ---
    bottom = np.zeros(n_rows)
    for j in range(n_cols):
        vals = pn['mat_pct'][:, j]
        bars = ax.bar(x, vals, bar_w, bottom=bottom,
                      color=pn['colors'][j], edgecolor='white',
                      linewidth=0.4, zorder=2)
        for bi, (bar_rect, v_pct, v_cnt) in enumerate(zip(bars, vals, pn['mat_count'][:, j])):
            if v_pct >= 12:
                bar_color = pn['colors'][j]
                txt_color = 'white' if bar_color == C3_DARK else TEXT_CLR
                ax.text(bar_rect.get_x() + bar_rect.get_width() / 2,
                        bar_rect.get_y() + bar_rect.get_height() / 2,
                        str(v_cnt), ha='center', va='center',
                        fontsize=7.5, color=txt_color)
        bottom += vals

    # --- Row totals above bars ---
    for xi, row_total in enumerate(pn['mat_count'].sum(axis=1)):
        ax.text(xi, 107, f'$n$={row_total}', ha='center', va='center',
                fontsize=7.5, color=TEXT_CLR)

    # --- p-value above n=XX labels, centered in panel ---
    p_val = pn['p']
    if p_val < 0.001:
        p_text = r'$P$ < 0.001'
    else:
        p_trunc = int(p_val * 1000) / 1000
        p_text = f'$P$ = {p_trunc:.3f}'
    ax.text((n_rows - 1) / 2, 114, p_text, ha='center', va='center',
            fontsize=8, color=TEXT_CLR)

    # --- Axes formatting ---
    ax.set_xlim(-0.65, n_rows - 0.35)
    ax.set_ylim(0, 120)
    ax.set_xticks(x)
    ax.set_xticklabels(pn['row_labels'], fontsize=8, color=TEXT_CLR)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50', '75', '100'], fontsize=7.5, color=TEXT_CLR)
    if i == 0:
        ax.set_ylabel('Percentage (%)', fontsize=8.5, color=TEXT_CLR, labelpad=2)

    # --- Spines ---
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color(TEXT_CLR)
    ax.tick_params(width=0.6, length=3)

    # --- Panel label ---
    ax.text(-0.12, 1.06, LABELS[i], transform=ax.transAxes,
            ha='left', va='top', fontsize=10.5, fontweight='bold',
            color=TEXT_CLR)

    # --- Legend ---
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec='white', lw=0.4)
                      for c in pn['colors']]
    ncl = pn['ncol_legend']
    leg = ax.legend(
        handles=legend_handles,
        labels=pn['col_labels'],
        loc='upper center',
        bbox_to_anchor=(0.50, -0.19),
        fontsize=7, frameon=True, edgecolor='#cccccc',
        framealpha=1, facecolor='white',
        ncol=ncl, labelspacing=0.25, handlelength=0.9,
        handletextpad=0.5, columnspacing=1.0, borderpad=0.35,
        title=pn['legend_title'],
        title_fontsize=7,
    )
    leg.get_title().set_color(TEXT_CLR)

# ==================== SAVE ====================

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'Figure_3.jpg')
fig.savefig(out, dpi=DPI, format='jpeg', facecolor=BG, edgecolor='none', pil_kwargs={'quality': 95})
plt.close()

print(f'Saved {out}')
print(f'Panel A: Kendall τ = {tau_a:.3f}, p = {p_a:.4f}')
print(f'Panel B: Kendall τ = {tau_b:.3f}, p = {p_b:.4f}')
print(f'Panel C: Kendall τ = {tau_c:.3f}, p = {p_c:.4f}')
