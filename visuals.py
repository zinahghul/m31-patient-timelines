import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# =========================================================
# Figure
# =========================================================
fig, ax = plt.subplots(figsize=(20, 6))

ax.set_xlim(0, 20)
ax.set_ylim(0, 6)
ax.axis("off")

# =========================================================
# Colors
# =========================================================
blue = "#2563EB"
teal = "#0F766E"
purple = "#7C3AED"
orange = "#EA580C"
red = "#DC2626"
green = "#15803D"
gray = "#6B7280"
dark = "#111827"

# =========================================================
# Box settings
# =========================================================
box_y = 2.15
box_h = 1.65

# x position, width, title, subtitle, color
boxes = [
    (0.20, 2.65, "Patient\nTimeline",
     "Chronological\nEHR events", blue),

    (3.25, 2.90, "Clinical Tokens",
     "COND • ENC • ALG • CARE\n+ temporal gaps", teal),

    (6.55, 2.30, "Embedding",
     "Token + position", purple),

    (9.20, 2.70, "Transformer",
     "3 layers • 4 heads", orange),

    (12.25, 2.20, "[CLS]",
     "Patient\nrepresentation", red),

    (14.85, 4.70, "40-Condition\nPrediction",
     "Multi-label output", green)
]



# =========================================================
# Draw boxes
# =========================================================
for x, width, title, subtitle, color in boxes:

    box = FancyBboxPatch(
        (x, box_y),
        width,
        box_h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=2.2,
        edgecolor=color,
        facecolor=color,
        alpha=0.10
    )

    ax.add_patch(box)

    # Title
    ax.text(
        x + width / 2,
        box_y + 1.18,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=color,
        linespacing=1.25
    )

    # Subtitle
    ax.text(
        x + width / 2,
        box_y + 0.38,
        subtitle,
        ha="center",
        va="center",
        fontsize=11.5,
        color=dark,
        linespacing=1.5
    )



# =========================================================
# Arrows
# =========================================================
arrow_y = box_y + box_h / 2

for i in range(len(boxes) - 1):

    x1, w1 = boxes[i][0], boxes[i][1]
    x2 = boxes[i + 1][0]

    arrow = FancyArrowPatch(
        (x1 + w1 + 0.08, arrow_y),
        (x2 - 0.08, arrow_y),
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=2.2,
        color=gray
    )

    ax.add_patch(arrow)


# =========================================================
# Title
# =========================================================
ax.text(
    10,
    5.45,
    "EHRTransformer Architecture",
    ha="center",
    va="center",
    fontsize=25,
    fontweight="bold",
    color=dark
)

ax.text(
    10,
    5.03,
    "Leakage-controlled longitudinal patient timeline forecasting",
    ha="center",
    va="center",
    fontsize=13,
    color=gray
)


# =========================================================
# Bottom summary
# =========================================================
ax.text(
    10,
    1.05,
    "128-d embeddings   •   512-token maximum sequence   •   40-condition multi-label output",
    ha="center",
    va="center",
    fontsize=12.5,
    color=gray
)


# =========================================================
# Save
# =========================================================
plt.savefig(
    "Figure_1_EHRTransformer_Architecture.png",
    dpi=400,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    "Figure_1_EHRTransformer_Architecture.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.show()