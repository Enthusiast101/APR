import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress


# ============================================================
# Authoritative Table 9 data
# ============================================================

data = [
    # model, domain, delta_RR, recover, harden, unlearn
    ("Llama-3.1-8B", "Math",     0.019, 0.971, 0.947, 0.938),
    ("Llama-3.1-8B", "Code",     0.274, 0.768, 0.955, 0.986),
    ("Llama-3.1-8B", "Medical",  0.255, 0.607, 0.840, 0.704),
    ("Llama-3.1-8B", "Legal",    0.277, 0.753, 0.945, 0.829),
    ("Llama-3.1-8B", "Dolly",    0.401, 0.486, 0.838, 0.650),

    ("Llama-3.2-3B", "Math",     0.049, 0.856, 0.945, 0.890),
    ("Llama-3.2-3B", "Code",     0.625, 0.738, 0.907, 0.992),
    ("Llama-3.2-3B", "Medical",  0.172, 0.756, 0.908, 0.673),
    ("Llama-3.2-3B", "Legal",    0.360, 0.652, 0.458, 0.503),
    ("Llama-3.2-3B", "Dolly",    0.453, 0.663, 0.912, 0.580),

    ("Gemma-3-4B",   "Math",     0.004, 0.721, 0.653, 0.778),
    ("Gemma-3-4B",   "Code",     0.209, 0.547, 0.658, 0.940),
    ("Gemma-3-4B",   "Dolly",    0.263, 0.481, 0.655, 0.617),

    ("Qwen3-4B",     "Math",     0.023, 0.809, 0.682, 0.829),
    ("Qwen3-4B",     "Code",     0.245, 0.510, 0.671, 0.806),
    ("Qwen3-4B",     "Dolly",    0.291, 0.451, 0.674, 0.458),
]


models = np.array([row[0] for row in data])
domains = np.array([row[1] for row in data])

x = np.array([row[2] for row in data])
recover = np.array([row[3] for row in data])
harden = np.array([row[4] for row in data])
unlearn = np.array([row[5] for row in data])


# ============================================================
# Plot configuration
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

methods = {
    "RECOVER": recover,
    "HARDEN": harden,
    "UNLEARN": unlearn,
}

markers = {
    "RECOVER": "o",
    "HARDEN": "D",
    "UNLEARN": "^",
}

# Use explicit colors only if you want a paper-style palette.
# Otherwise matplotlib's default cycle can be used.
colors = {
    "RECOVER": "tab:red",
    "HARDEN": "tab:brown",
    "UNLEARN": "tab:green",
}


# ============================================================
# Plot individual observations
# ============================================================

for method, y in methods.items():

    ax.scatter(
        x,
        y,
        label=method,
        marker=markers[method],
        color=colors[method],
        s=65,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )


# ============================================================
# Regression lines
#
# IMPORTANT:
# These are statistical trend lines, NOT regime boundaries.
# ============================================================

x_line = np.linspace(0, 0.65, 300)

for method, y in methods.items():

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    y_line = intercept + slope * x_line

    ax.plot(
        x_line,
        y_line,
        color=colors[method],
        linestyle="--",
        linewidth=1.8,
        alpha=0.8,
    )

    # Regression statistics
    print(
        f"{method:8s}: "
        f"slope={slope:.4f}, "
        f"r={r_value:.3f}, "
        f"p={p_value:.3f}"
    )


# ============================================================
# Regime boundaries from the original hypothesis
# ============================================================

ax.axvline(
    0.10,
    color="gray",
    linestyle=":",
    linewidth=1.5,
    alpha=0.8,
)

ax.axvline(
    0.35,
    color="gray",
    linestyle=":",
    linewidth=1.5,
    alpha=0.8,
)


# ============================================================
# Regime labels
# ============================================================

ax.text(
    0.05,
    1.02,
    "Mild",
    ha="center",
    va="bottom",
    transform=ax.get_xaxis_transform(),
)

ax.text(
    0.225,
    1.02,
    "Broad",
    ha="center",
    va="bottom",
    transform=ax.get_xaxis_transform(),
)

ax.text(
    0.50,
    1.02,
    "Severe",
    ha="center",
    va="bottom",
    transform=ax.get_xaxis_transform(),
)


# ============================================================
# Annotate every point
# ============================================================

for i in range(len(data)):

    # Short labels for readability
    model_short = {
        "Llama-3.1-8B": "L8",
        "Llama-3.2-3B": "L3",
        "Gemma-3-4B": "G4",
        "Qwen3-4B": "Q4",
    }[models[i]]

    domain_short = {
        "Math": "math",
        "Code": "code",
        "Medical": "med",
        "Legal": "legal",
        "Dolly": "dolly",
    }[domains[i]]

    label = f"{model_short}-{domain_short}"

    # The flagged Llama-3.2-3B/legal cell
    if models[i] == "Llama-3.2-3B" and domains[i] == "Legal":
        label += " *"

    # Put annotation slightly above the highest of the three
    y_max = max(
        recover[i],
        harden[i],
        unlearn[i],
    )

    ax.annotate(
        label,
        (x[i], y_max),
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=7.5,
        alpha=0.85,
    )


# ============================================================
# Axes
# ============================================================

ax.set_xlim(0, 0.65)
ax.set_ylim(0.4, 1.03)

ax.set_xlabel(
    r"Drift severity $\Delta RR = \overline{RR}_{aligned}"
    r" - \overline{RR}_{drifted}$",
    fontsize=11,
)

ax.set_ylabel(
    r"Post-repair refusal rate $\overline{RR}$",
    fontsize=11,
)

ax.set_title(
    "Post-repair refusal rate vs. drift severity",
    fontsize=13,
    pad=18,
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.25,
)

ax.legend(
    loc="lower left",
    frameon=True,
)


# ============================================================
# Flag the questionable cell explicitly
# ============================================================

ax.text(
    0.625,
    0.415,
    "* Llama-3.2-3B/legal: data-quality flag",
    ha="right",
    va="bottom",
    fontsize=8,
)


plt.tight_layout()
plt.show()