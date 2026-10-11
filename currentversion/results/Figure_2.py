from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

# ==================== PARAMETERS ====================
FIG_W_MM = 180
FIG_H_MM = 95
DPI = 300
FONT = 'Arial'
TEXT = '#252525'
BACKGROUND = '#ffffff'
CRUDE = '#62A9C8'
ADJUSTED = '#147EBC'
REFERENCE = '#A8D1E1'
FACTORS = [
    ('Education', 'Education\n(per category increase)'),
    ('Annual household income', 'Annual household income\n(per category increase)'),
    ('Time since first dental treatment', 'Time since first dental treatment\n(per category increase)'),
    ('Regular dental checkups', 'Regular dental checkups\n(yes vs no)'),
    ('First periodontal surgery (Q7)', 'First periodontal surgery\n(yes vs no)'),
]
RESULTS = Path(__file__).with_name('regression_results.json')
OUTPUT = RESULTS.parent.parent / 'figures' / 'Figure_2.jpg'

# ==================== DATA ====================
models = json.loads(RESULTS.read_text())['models']
plt.rcParams.update({
    'font.family': FONT, 'font.size': 9, 'text.color': TEXT,
    'axes.labelcolor': TEXT, 'axes.edgecolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
})
fig, ax = plt.subplots(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
                       dpi=DPI, facecolor=BACKGROUND)
fig.subplots_adjust(left=0.285, right=0.505, top=0.89, bottom=0.22)
ax.set_facecolor(BACKGROUND)
ax.axvline(1, color=REFERENCE, linewidth=0.9, linestyle='--')
y_positions = np.arange(len(FACTORS))[::-1]
ax.set_ylim(-0.55, len(FACTORS) - 0.4)
ax.set_xscale('log')
ax.set_xlim(0.25, 8)

# ==================== DRAW ====================
for y, (name, label) in zip(y_positions, FACTORS):
    model = models[name]
    for result, color, offset, filled in [
        (model['Crude']['MLE'], CRUDE, 0.13, False),
        (model['MLE'], ADJUSTED, -0.13, True),
    ]:
        e = result['effects'][0]
        ax.plot([e['CI_low'], e['CI_high']], [y + offset] * 2,
                color=color, linewidth=1.7, solid_capstyle='round')
        ax.plot(e['OR'], y + offset, marker='o', markersize=5.5,
                markerfacecolor=color if filled else BACKGROUND,
                markeredgecolor=color, markeredgewidth=1.1)
    y_fraction = fig.transFigure.inverted().transform(ax.transData.transform((1, y)))[1]
    for column, result in [(0.53, model['Crude']['MLE']), (0.765, model['MLE'])]:
        e = result['effects'][0]
        fig.text(column, y_fraction,
                 f"{e['OR']:.2f} ({e['CI_low']:.2f}–{e['CI_high']:.2f})",
                 ha='left', va='center', fontsize=8.8)

# ==================== AXES ====================
ax.set_yticks(y_positions, [label for _, label in FACTORS])
ax.tick_params(axis='y', length=0, pad=7, labelsize=8.5)
ax.set_xticks([0.25, 0.5, 1, 2, 4, 8], ['0.25', '0.5', '1', '2', '4', '8'])
ax.xaxis.set_minor_locator(NullLocator())
ax.tick_params(axis='x', length=3, width=0.7, labelsize=8)
ax.set_xlabel('Odds ratio', labelpad=5)
for side in ['top', 'right', 'left']:
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_linewidth(0.7)
fig.text(0.53, 0.94, 'Crude OR (95% CI)', fontsize=9, ha='left')
fig.text(0.765, 0.94, 'Adjusted OR (95% CI)', fontsize=9, ha='left')
fig.legend(handles=[
    Line2D([0], [0], color=CRUDE, marker='o', markerfacecolor=BACKGROUND,
           markersize=5.5, linewidth=1.7, label='Crude'),
    Line2D([0], [0], color=ADJUSTED, marker='o', markersize=5.5,
           linewidth=1.7, label='Age- and sex-adjusted'),
], loc='lower center', bbox_to_anchor=(0.62, 0.025), ncol=2,
    frameon=False, fontsize=9)

# ==================== SAVE ====================
OUTPUT.parent.mkdir(exist_ok=True)
fig.savefig(OUTPUT, dpi=DPI, format='jpeg', facecolor=BACKGROUND,
            pil_kwargs={'quality': 95, 'subsampling': 0})
plt.close(fig)
print(f'Saved {OUTPUT}')
