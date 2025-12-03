import matplotlib.pyplot as plt
import numpy as np

# x-axis: the “shots” you have
shots_labels = ["0", "1", "2", "4", "8", "all"]
x = np.arange(len(shots_labels))      # positions on the axis: 0..5

# ==== FILL THESE WITH YOUR RESULTS ====
# One list per method, same length as shots_labels
results = {
    "Method A": [60.1, 72.3, 78.5, 85.2, 90.0, 93.5],
    "Method B": [55.0, 70.0, 77.2, 84.0, 89.1, 92.0],
    "Method C": [50.0, 68.0, 75.0, 82.0, 88.0, 91.0],
    # add more methods if you like
}
# ======================================

fig, ax = plt.subplots(figsize=(5, 4))

# light gray background like in your example
ax.set_facecolor("#f0f0f0")

# plot one colored line per method
markers = ["o", "s", "^", "D", "v", "P"]  # different marker shapes
for (i, (name, scores)) in enumerate(results.items()):
    ax.plot(
        x,
        scores,
        marker=markers[i % len(markers)],
        linewidth=2,
        label=name,
    )

# x-axis ticks & labels
ax.set_xticks(x)
ax.set_xticklabels(shots_labels)
ax.set_xlabel("Number of labeled training examples per class")
ax.set_ylabel("Score (%)")
ax.set_title("Flowers102")

# horizontal grid lines help readability
ax.grid(axis="y", linestyle="--", alpha=0.4)

# legend box like in your screenshot
ax.legend(framealpha=0.9)

plt.tight_layout()
plt.show()
