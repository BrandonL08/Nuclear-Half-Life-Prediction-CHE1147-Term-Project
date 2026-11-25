"""
EDA for extracted and cleaned ENSDF-derived nuclear data.

Outputs: figures saved to EDA/figures/ and displayed on screen.
The extracted half-life is plotted in seconds on a logarithmic y-axis;
cleaned log_half_life is plotted as stored (its log base is not assumed).
"""

# %% Imports and settings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

EDA_DIR = Path.cwd()
EXTRACTED_CSV = EDA_DIR.parent / "Extraction" / "extracted_data.csv"
CLEANED_CSV = EDA_DIR / "cleaned_data.csv"
FIGURE_DIR = EDA_DIR / "figures"
SAVE_FIGURES = True
SHOW_PLOTS = True

COLORS = {"Extracted": "#3e4989", "Cleaned": "#6ece58"}
VIRIDIS = plt.get_cmap("viridis")
KEV_FORMAT = FuncFormatter(lambda value, _: f"{value:.1e}")
CORRELATION_FEATURES = ["A", "N", "Z", "N/Z", "Q_alpha_keV", "Q_beta_keV", "S_n_keV", "S_p_keV"]
ENERGY_FEATURES = {
    "Q_alpha_keV": r"$\mathrm{Q}_{\alpha}$",
    "Q_beta_keV": r"$\mathrm{Q}_{\beta}$",
    "S_n_keV": r"$\mathrm{S}_{\mathrm{n}}$",
    "S_p_keV": r"$\mathrm{S}_{\mathrm{p}}$",
}
SCATTER_FEATURES = {
    "N/Z": "Neutron/Proton Ratio - N/Z",
    "A": "Mass number - A",
    **{feature: f"{label}" for feature, label in ENERGY_FEATURES.items()},
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "black",
    "grid.linestyle": ":",
    "grid.linewidth": 0.35,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_data(path, target):
    """Load a dataset and prepare numeric columns without deleting rows."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = {"A", "Z", target, *ENERGY_FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    numeric = ["A", "Z", "N", target, *ENERGY_FEATURES]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "N" not in df.columns:
        df["N"] = df["A"] - df["Z"]
    df["N/Z"] = df["N"] / df["Z"].where(df["Z"] > 0)
    return df


def numeric_values(df, feature):
    """Finite values for one feature, excluding only missing/invalid entries."""
    values = df[feature].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def finish_plot(fig, filename):
    if SAVE_FIGURES:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / filename, dpi=250, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# %% Scatter Plots

def plot_scatter_grid(df, label, target):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    y = df[target]

    for ax, (feature, xlabel) in zip(axes.flat, SCATTER_FEATURES.items()):
        x = df[feature]
        valid = np.isfinite(x) & np.isfinite(y)
        if target == "half_life_seconds":
            valid &= y > 0  # Logarithmic axes cannot show zero/negative half-lives.

        ax.scatter(x[valid], y[valid], s=13,
                   color=COLORS[label], edgecolors="none", rasterized=True)
        if target == "half_life_seconds":
            ax.set_yscale("log")
        
        axis_label = f"{xlabel} (keV)" if feature in ENERGY_FEATURES else feature
        ax.set(xlabel=axis_label, title=xlabel)
        
        if feature in ENERGY_FEATURES:
            ax.xaxis.set_major_formatter(KEV_FORMAT)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="x", labelsize=9, labelrotation=0)
        ax.text(0.98, 0.96, f"n = {valid.sum():,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="black")
        ax.grid(False)

    y_label = ("Half-life (s; log scale)" if target == "half_life_seconds"
               else "log_half_life (as stored)")
    fig.suptitle(f"{label} data | Feature relationships with half-life",
                 fontsize=16, fontweight="bold")
    fig.supylabel(y_label, x=0.01, fontsize=11)
    fig.tight_layout(rect=(0.035, 0.01, 1, 0.96), h_pad=3.0, w_pad=3.8)
    finish_plot(fig, f"scatter_{label.lower()}.png")


# %% Boxplots

def plot_boxplots(df1, df2):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (feature, title) in zip(axes.flat, ENERGY_FEATURES.items()):
        groups = [numeric_values(df1, feature), numeric_values(df2, feature)]
        available = [(i, values) for i, values in enumerate(groups, 1) if len(values)]
        if available:
            bp = ax.boxplot([values for _, values in available],
                            positions=[i for i, _ in available], widths=0.48,
                            patch_artist=True, showfliers=False,
                            medianprops={"color": "black", "linewidth": 2},
                            whiskerprops={"color": "black"},
                            capprops={"color": "black"})
            for patch, (i, _) in zip(bp["boxes"], available):
                color = list(COLORS.values())[i - 1]
                patch.set(facecolor=color, edgecolor=color)
        else:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                    ha="center", va="center")
        ax.set_xticks([1, 2], [f"Extracted\n(n={len(groups[0]):,})",
                               f"Cleaned\n(n={len(groups[1]):,})"])
        ax.set(xlim=(0.45, 2.55), title=title, ylabel="Energy (keV)")
        ax.yaxis.set_major_formatter(KEV_FORMAT)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y")

    fig.suptitle("Q and S distributions | Boxplots", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.015, "Outlier markers hidden for legibility; underlying data are unchanged.",
             ha="center", fontsize=9, color="black")
    fig.tight_layout(rect=(0, 0.035, 1, 0.955), h_pad=2)
    finish_plot(fig, "boxplots_Q_S.png")


# %% Histograms

def plot_histograms(df1, df2):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (feature, title) in zip(axes.flat, ENERGY_FEATURES.items()):
        groups = [("Extracted", numeric_values(df1, feature)),
                  ("Cleaned", numeric_values(df2, feature))]
        available = [values for _, values in groups if len(values)]
        if available:
            # Shared bin edges and density scaling make sample sizes comparable.
            bins = np.histogram_bin_edges(np.concatenate(available), bins=35)
            for label, values in groups:
                if len(values):
                    ax.hist(values, bins=bins, density=True, histtype="step",
                            linewidth=2, color=COLORS[label],
                            label=f"{label} (n={len(values):,})")
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                    ha="center", va="center")
        ax.set(xlabel="Energy (keV)", ylabel="Density", title=title)
        ax.xaxis.set_major_formatter(KEV_FORMAT)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="x", labelsize=9, labelrotation=0)
        # Explicit tick labels avoid Matplotlib's floating 1e-5 offset.
        ax.yaxis.set_major_formatter(KEV_FORMAT)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y")

    fig.suptitle("Q and S distributions | Histograms", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.01, 1, 0.955), h_pad=2.4)
    finish_plot(fig, "histograms_Q_S.png")


# %% Feature Coverage

def plot_coverage(df1, df2):
    columns = [
        ("A", "A", "A"),
        *[(feature, feature, label) for feature, label in ENERGY_FEATURES.items()],
    ]
    coverage1 = [100 * np.isfinite(df1[raw]).mean() for raw, _, _ in columns]
    coverage2 = [100 * np.isfinite(df2[clean]).mean() for _, clean, _ in columns]
    x = np.arange(len(columns))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.3))
    for offset, label, values in [(-width / 2, "Extracted", coverage1),
                                   (width / 2, "Cleaned", coverage2)]:
        bars = ax.bar(x + offset, values, width, color=COLORS[label],
                      label=label)
        ax.bar_label(bars, labels=[f"{v:.0f}%" for v in values],
                     padding=3, fontsize=8, color="black")

    ax.set_xticks(x, [name for _, _, name in columns])
    ax.set(ylim=(0, 110), ylabel="Available observations (%)",
           title="Feature Coverage | Extracted vs. Cleaned")
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    finish_plot(fig, "feature_coverage.png")


# %% Half-life Status (Extracted Data)

def plot_half_life_status(df1):
    if "half_life_status" not in df1.columns:
        print("Half-life status unavailable: extracted data has no 'half_life_status' column.")
        return

    status = (df1["half_life_status"].astype("string").str.strip()
              .fillna("Missing").replace("", "Missing").value_counts())
    print("\nExtracted half-life status:\n", status.to_string())
    fig, ax = plt.subplots(figsize=(9, max(3.6, 0.55 * len(status) + 1.6)))
    shades = VIRIDIS(np.linspace(0.22, 0.78, len(status)))
    bars = ax.barh(status.index[::-1], status.values[::-1], color=shades[::-1])
    for bar, count in zip(bars, status.values[::-1]):
        ax.text(bar.get_width() + max(len(df1) * 0.012, 0.2),
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}  ({count / len(df1):.1%})", va="center", fontsize=9)
    ax.set(xlabel="Number of nuclides", title="Half-life status | Extracted data",
           xlim=(0, max(status.values) * 1.30 if len(status) else 1))
    ax.grid(axis="x")
    fig.tight_layout()
    finish_plot(fig, "half_life_status.png")


# %% Pearson Correlation Heatmaps

def plot_correlation_heatmaps(df1, df2):
    """Compare pairwise Pearson correlations using DataFrame.corr()."""
    labels = ["A", "N", "Z", "N/Z", r"$\mathrm{Q}_{\alpha}$", r"$\mathrm{Q}_{\beta}$", r"$\mathrm{S}_{\mathrm{n}}$", r"$\mathrm{S}_{\mathrm{p}}$"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)

    for ax, (name, df) in zip(axes, [("Extracted", df1), ("Cleaned", df2)]):
        corr = df[CORRELATION_FEATURES].corr()  # Pairwise-complete Pearson correlation.
        matrix = np.ma.masked_invalid(corr.to_numpy())
        image = ax.imshow(matrix, cmap="viridis", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(labels)), labels, rotation=0)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(f"{name} | Pearson correlation", fontsize=13, fontweight="bold", pad=15)
        ax.tick_params(length=0)
        for (row, col), value in np.ndenumerate(corr.to_numpy()):
            if np.isfinite(value):
                # Text contrasts with the viridis background.
                rgb = VIRIDIS((value + 1) / 2)[:3]
                text_color = "black" if np.dot(rgb, [0.2126, 0.7152, 0.0722]) > 0.50 else "white"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        color=text_color, fontsize=8)
    fig.colorbar(image, ax=axes, shrink=0.76, pad=0.03, label="Pearson correlation (r)")
    finish_plot(fig, "correlation_heatmaps.png")


# %% Run analysis

def main():
    df1 = load_data(EXTRACTED_CSV, "half_life_seconds")
    df2 = load_data(CLEANED_CSV, "log_half_life")
    print(f"Extracted (df1): {len(df1):,} rows x {df1.shape[1]} columns")
    print(f"Cleaned   (df2): {len(df2):,} rows x {df2.shape[1]} columns")
    for label, df in [("Extracted", df1), ("Cleaned", df2)]:
        print(f"\n{label} Q/S summary (keV):")
        print(df[list(ENERGY_FEATURES)].describe().round(1).T.to_string())

    plot_scatter_grid(df1, "Extracted", "half_life_seconds")
    plot_scatter_grid(df2, "Cleaned", "log_half_life")
    plot_boxplots(df1, df2)
    plot_histograms(df1, df2)
    plot_coverage(df1, df2)
    plot_half_life_status(df1)
    plot_correlation_heatmaps(df1, df2)


if __name__ == "__main__":
    main()