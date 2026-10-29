import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
SURF, INK, INK2, GRID = '#ffffff', '#0b0b0b', '#52514e', '#e4e4e0'
BLUE_SEQ = ['#ffffff', '#dbe8f8', '#a9c9ef', '#6fa3e3', '#2a78d6', '#1a56a3']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2})

# ---- Slide 14: confusion matrix, rows = adjudicated human label, cols = verifier
labels = ['SUPPORTED', 'UNSUPPORTED', 'CONTRADICTED']
M = np.array([[5, 2, 0], [0, 2, 1], [0, 0, 0]])
fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=200, facecolor=SURF)
for r in range(3):
    for c in range(3):
        v = M[r, c]
        col = BLUE_SEQ[min(v, 5)] if v else '#f6f6f4'
        ax.add_patch(plt.Rectangle((c + 0.03, 2 - r + 0.03), 0.94, 0.94, color=col, lw=0))
        txt_col = '#ffffff' if v >= 4 else INK
        ax.text(c + 0.5, 2 - r + 0.5, str(v), ha='center', va='center', fontsize=18 if (r == c and v) else 15,
                color=txt_col, fontweight='bold' if (r == c and v) else 'normal')
ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect('equal')
ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(['SUP', 'UNSUP', 'CONTRA'], color=INK)
ax.set_yticks([2.5, 1.5, 0.5]); ax.set_yticklabels(['SUP (7)', 'UNSUP (3)', 'CONTRA (0)'], color=INK)
ax.set_xlabel('Verifier prediction'); ax.set_ylabel('Human label (adjudicated)')
ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
fig.text(0.02, 0.02, 'n = 10 claims (stratified pilot). Diagonal = agreement: 7/10.', color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig('A_annotation_confusion.png', facecolor=SURF); plt.close(fig)

# ---- Slide 13: A prompt v1 -> v2 (dev, same 50 questions, same drafts); two panels, one measure each
BLUE, ORANGE = '#2a78d6', '#eb6834'   # v1 = orange, v2 = blue (identity, fixed order)
panels = [('Final answer written as a sentence', {'M2': (10, 0), 'M3': (6, 2)}),
          ('Critic flags a CORRECT draft', {'M2': (67, 41), 'M3': (89, 63)})]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=200, facecolor=SURF)
for ax, (title, data) in zip(axes, panels):
    x = np.arange(2); w = 0.36
    for k, (name, col, off) in enumerate((('v1 prompts', ORANGE, -w/2 - 0.01), ('v2 prompts', BLUE, w/2 + 0.01))):
        vals = [data[m][k] for m in ('M2', 'M3')]
        ax.bar(x + off, vals, width=w, color=col, label=name)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + max(max(d) for d in data.values())*0.03, f'{v}%', ha='center', color=INK, fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(['M2 self-reflection', 'M3 critic'], color=INK, fontsize=10)
    ymax = max(max(v) for v in data.values()); ax.set_ylim(0, ymax * 1.25 + 2)
    ax.set_title(title, loc='left', color=INK, fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
axes[0].legend(frameon=False, fontsize=9, loc='upper right', labelcolor=INK)
fig.text(0.01, 0.01, 'Dev set, 50 questions, identical shared drafts; only A\'s prompts differ.', color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig('A_v1_v2_revision_fix.png', facecolor=SURF); plt.close(fig)
