import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==================== PARAMETERS ====================

BG        = '#ffffff'
TEXT_CLR  = '#252525'
LINE_CLR  = '#252525'
BOX_FACE  = '#ffffff'
BOX_EDGE  = '#252525'

FIG_W_MM    = 180
FIG_H_MM    = 152
DPI         = 300
TARGET_W_PX = 2126

# ==================== DATA ====================

box1_title = 'Patients who received periodontal surgery'
box1_sub   = '(2022.07 – 2025.12)'
box1_n     = '$n$ = 2,876'

incl_head = 'Not meeting inclusion criteria:'
incl_list = [
    'Not Miller Class I/II gingival recession',
    'Did not receive mucogingival surgery',
    'Age < 18 years',
    'Unable to complete questionnaire independently',
    'Did not sign informed consent',
]

excl_head = 'Meeting exclusion criteria:'
excl_list = [
    'Severe uncontrolled systemic disease',
    'Mental illness or cognitive impairment',
    'Pregnant or lactating women',
    'Prior history of mucogingival surgery',
]

total_excl1 = 'Total excluded $n$ = 2,804'

box2_title = 'Invited patients who submitted questionnaires'
box2_n     = '$n$ = 72'

qc_head = 'Excluded (questionnaire quality control):'
qc_list = [
    'Completion time ≤ 120 s',
    'Incorrect information',
    'Highly similar response content',
]

total_excl2 = '$n$ = 6'

box3_title = 'Valid questionnaires included in final analysis'
box3_n     = '$n$ = 66'

# ==================== SETUP ====================

plt.rcParams['font.family'] = 'Arial'

fig_w = TARGET_W_PX / DPI
fig_h = FIG_H_MM / 25.4
fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG, dpi=DPI)
ax = fig.add_axes([0.025, 0.015, 0.95, 0.97], facecolor=BG)
ax.axis('off')

# ==================== GEOMETRY ====================

bx, bw, bh = 0.035, 0.55, 0.120
arrow_x = bx + bw / 2
br_x    = bx + bw + 0.035
tx_x    = br_x + 0.022

# Y centers
y1 = 0.894   # Box 1
y2 = 0.467   # Box 2
y3 = 0.221   # Box 3

# ==================== DRAW FUNCTIONS ====================

def draw_box(ax, x, y, w, h, title, subtitle, n_text):
    rect = mpatches.FancyBboxPatch(
        (x, y - h/2), w, h,
        boxstyle='round,pad=0.022',
        facecolor=BOX_FACE, edgecolor=BOX_EDGE,
        linewidth=1.1, zorder=3)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + 0.014, title, ha='center', va='bottom',
                fontsize=10.5, color=TEXT_CLR, zorder=4)
        ax.text(x + 0.025, y - 0.012, subtitle, ha='left', va='top',
                fontsize=9, color=TEXT_CLR, zorder=4)
        ax.text(x + w - 0.025, y - 0.012, n_text, ha='right', va='top',
                fontsize=9.5, color=TEXT_CLR, zorder=4)
    else:
        ax.text(x + 0.025, y, title, ha='left', va='center',
                fontsize=10.5, color=TEXT_CLR, zorder=4)
        ax.text(x + w - 0.025, y, n_text, ha='right', va='center',
                fontsize=9.5, color=TEXT_CLR, zorder=4)


def draw_arrow_down(ax, x, y0, y1):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color=LINE_CLR, lw=1.5),
                zorder=1)


def calc_bracket_height(sections, bracket_gap=0.016):
    """Calculate total bracket text height in data coords."""
    h = 0.018
    for si, sec in enumerate(sections):
        h += 0.026
        h += len(sec['items']) * 0.023
        if si < len(sections) - 1:
            h += bracket_gap
    h += 0.012
    return h


def draw_excl_bracket(ax, br_x, tx_x, y_mid, sections, total_n=None):
    """Draw exclusion bracket centered at y_mid, sized to content."""
    bracket_h = calc_bracket_height(sections)
    y_top = y_mid + bracket_h / 2
    y_bot = y_mid - bracket_h / 2

    lw = 0.8
    ax.plot([br_x, tx_x], [y_bot, y_bot], color=LINE_CLR, linewidth=lw, clip_on=False)
    ax.plot([br_x, tx_x], [y_top, y_top], color=LINE_CLR, linewidth=lw, clip_on=False)
    ax.plot([br_x, br_x], [y_bot, y_top], color=LINE_CLR, linewidth=lw, clip_on=False)

    y_pos = y_top - 0.018

    for si, sec in enumerate(sections):
        head = sec['head']
        items = sec['items']

        ax.text(tx_x, y_pos, head, ha='left', va='top',
                fontsize=8.5, color=TEXT_CLR, clip_on=False)
        y_pos -= 0.026

        for item in items:
            if y_pos < y_bot:
                break
            ax.text(tx_x + 0.006, y_pos, f'• {item}',
                    ha='left', va='center', fontsize=7.5, color=TEXT_CLR, clip_on=False)
            y_pos -= 0.023

        if si < len(sections) - 1:
            y_pos -= 0.016

    if total_n:
        ax.text(tx_x, y_bot + 0.005, total_n, ha='left', va='bottom',
                fontsize=7.5, color=TEXT_CLR, clip_on=False)


# ==================== RENDER ====================

# Box 1
draw_box(ax, bx, y1, bw, bh, box1_title, box1_sub, box1_n)

# Gap 1 midpoint
b1_bot = y1 - bh/2
b2_top = y2 + bh/2
gap1_mid = (b1_bot + b2_top) / 2

# Arrow 1
draw_arrow_down(ax, arrow_x, b1_bot - 0.004, b2_top + 0.004)

# Exclusion bracket 1
draw_excl_bracket(ax, br_x, tx_x, gap1_mid, [
    {'head': incl_head, 'items': incl_list},
    {'head': excl_head, 'items': excl_list},
], total_n=total_excl1)

# Box 2
draw_box(ax, bx, y2, bw, bh, box2_title, None, box2_n)

# Gap 2
b2_bot = y2 - bh/2
b3_top = y3 + bh/2
gap2_mid = (b2_bot + b3_top) / 2

# Arrow 2
draw_arrow_down(ax, arrow_x, b2_bot - 0.004, b3_top + 0.004)

# Exclusion bracket 2
draw_excl_bracket(ax, br_x, tx_x, gap2_mid, [
    {'head': qc_head, 'items': qc_list},
], total_n=total_excl2)

# Box 3
draw_box(ax, bx, y3, bw, bh, box3_title, None, box3_n)

ax.set_xlim(0, 1)
ax.set_ylim(0.12, 0.98)

# ==================== SAVE ====================

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'Figure_1.png')
fig.savefig(out, dpi=DPI, format='png', facecolor=BG, edgecolor='none')
plt.close()
print(f'Saved {out}')
