import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- data you’ll probably tweak ---
rows          = ["A","B","C","D"]
reward_times  = [0.9, 1.9, 3.0, 9.0]   # seconds from odour cue for A..D
trial_pulse   = (-0.9, 0.45)           # (start, width) in seconds
odour_pulse   = (0.00, 0.28)           # (start, width) in seconds
xticks        = [-1.25, 0, 0.6, 1.5, 3.75, 9.375]
colors = {
    1: (171/255, 92/255, 154/255),
    2: (138/255, 110/255, 168/255),
    3: (136/255, 172/255, 215/255),
    4: (142/255, 206/255, 216/255),
}
# --- figure/axes ---
fig, ax = plt.subplots(figsize=(8,5))
ystep = 1.2
ys = [ystep*(len(rows)-1 - i) for i in range(len(rows))]  # top→bottom

# helper to draw a small “step” rectangle sitting on the row baseline
def pulse(ax, x0, y, width, height=0.35, color="k", lw=1.5):
    ax.add_patch(Rectangle((x0, y - height/2), width, height,
                           fill=False, edgecolor=color, linewidth=lw,
                           joinstyle="miter"))

# baselines and labels
for i, (label, y) in enumerate(zip(rows, ys)):
    ax.hlines(y, xmin=xticks[0], xmax=xticks[-1]+0.4, color="k", lw=1)
    ax.text(xticks[0]-0.55, y, label, va="center", ha="left",
            fontsize=12, color=["#b56bb7","#7c4aa8","#66aee0","#79d3d3"][i])

# pulses
for i, y in enumerate(ys):
    pulse(ax, trial_pulse[0], y, trial_pulse[1], color="k")
    pulse(ax, odour_pulse[0], y, odour_pulse[1],
          color=["#b56bb7","#7c4aa8","#66aee0","#79d3d3"][i])

# droplets (use an emoji for simplicity; replace with a custom Path for a true icon)
for t, y in zip(reward_times, ys):
    ax.text(t, y+0.02, "💧", fontsize=18, va="center", ha="center")

# x-axis with arrow + ticks/labels
ax.set_xlim(xticks[0], xticks[-1]+0.4)
ax.set_ylim(-0.6, ys[0]+0.8)
ax.set_xticks(xticks)
ax.set_xlabel("Time from odour cue (s)")
for spine in ["left","top","right"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_position(("data", -0.2))
ax.tick_params(axis="y", length=0)  # hide y ticks
# arrow
ax.annotate("", xy=(xticks[-1]+0.35, -0.2), xytext=(xticks[0], -0.2),
            arrowprops=dict(arrowstyle="->", linewidth=1.5))

# left title
ax.text(xticks[0]-0.9, ys[0]+0.55, "Odour", fontsize=12, rotation=90, ha="center", va="top")

# row captions above pulses
ax.text(-0.35, ys[0]+0.55, "Trial start", rotation=45, ha="left", va="bottom")
ax.text(0.10,  ys[0]+0.55, "Odour cue",  rotation=45, ha="left", va="bottom")
ax.text(3.6,   ys[0]+0.55, "Reward",     rotation=45, ha="left", va="bottom")

plt.tight_layout()
plt.show()
