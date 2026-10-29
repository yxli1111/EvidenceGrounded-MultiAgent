import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
SURF, INK, INK2, GRID = '#ffffff', '#0b0b0b', '#52514e', '#e4e4e0'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2})

# ---- Chart 1: EM change vs M1 with 95% paired bootstrap CI (E's official numbers, main_v2, n=100)
rows = [('M2  Self-reflection', 0.51, -0.12, -0.19, -0.05),
        ('M3  Independent critic', 0.55, -0.08, -0.17, 0.00)]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200, facecolor=SURF); ax.set_facecolor(SURF)
for i, (name, em, d, lo, hi) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle='round', zorder=2)
    ax.scatter([d], [y], s=90, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
    ax.text(hi + 0.012, y, f'{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]', va='center', color=INK, fontsize=11)
ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)))
ax.text(0.004, 1.6, 'M1 (EM 0.63)', color=INK2, fontsize=10, ha='left', va='center')
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{r[0]}\nEM {r[1]:.2f}' for r in rows][::-1], color=INK, fontsize=12)
ax.set_xlim(-0.25, 0.2); ax.set_ylim(-0.6, 1.8)
ax.set_xlabel('Change in EM vs M1 (95% paired bootstrap CI)')
ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
fig.suptitle('Revision without evidence lowers EM', x=0.02, ha='left', color=INK, fontsize=13, fontweight='bold')
fig.tight_layout(); fig.savefig('A_rq1_delta_em.png', facecolor=SURF); plt.close(fig)

# ---- Chart 2: revision flips (main_v2)
methods = ['M2  Self-reflection', 'M3  Independent critic']
fixed, broke = [2, 5], [14, 13]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200, facecolor=SURF); ax.set_facecolor(SURF)
h = 0.34
for i, m in enumerate(methods):
    y = len(methods) - 1 - i
    for off, val, col, lab in ((+h/2 + 0.02, broke[i], ORANGE, 'Correct → wrong'), (-h/2 - 0.02, fixed[i], BLUE, 'Wrong → correct')):
        ax.barh(y + off, val, height=h, color=col, label=lab if i == 0 else None)
        ax.text(val + 0.3, y + off, str(val), va='center', color=INK, fontsize=11)
ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods[::-1], color=INK, fontsize=12)
ax.set_xlim(0, 17); ax.set_xlabel('Questions (of 100)')
ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.legend(frameon=False, loc='lower right', fontsize=10, labelcolor=INK)
fig.suptitle('Revisions break more correct answers than they fix', x=0.02, ha='left', color=INK, fontsize=13, fontweight='bold')
ax.set_xticks(range(0, 17, 4))
fig.tight_layout(); fig.savefig('A_rq1_flips.png', facecolor=SURF); plt.close(fig)
