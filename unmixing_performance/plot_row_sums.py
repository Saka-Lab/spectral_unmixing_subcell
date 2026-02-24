import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

DESIRED_CHANNEL_ORDER = [
    "DAPI",
    "ATTO 425",
    "ATTO 488",
    "ATTO 490LS",
    "Oregon Green 514",
    "ALEXA 532",
    "ATTO 550",
    "Atto Rho11",
    "ALEXA 594",
    "ATTO 633",
    "ALEXA 647",
    "TYE 705",
    "ALEXA 750",
    "CF770"
]

def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return None

def plot_row_sums_boxplot_with_stats(
    normalized_long_csv,
    desired_channel_order=DESIRED_CHANNEL_ORDER,
    output_dir="plots",
    figsize=(11, 6),
    y_col="value_norm",
    y_label="Normalized Value",
    title="Normalized Values per Channel (ANOVA + Tukey)",
    output_name="boxplot_with_stats.png"
):
    
    df = pd.read_csv(normalized_long_csv)

    # Parse base_channel and comparison
    if df["channel"].str.contains(" - ").any():
        df["base_channel"] = df["channel"].str.split(" - ").str[0]
        df["comparison"]   = df["channel"].str.split(" - ").str[1]
    else:
        # Already separated row sums, need a comparison column
        df["base_channel"] = df["channel"]
        df["comparison"]   = df["comparison"]  # already exists

    # -------------------------------
    # Enforce categorical types FIRST
    # -------------------------------
    comparison_order = ["raw", "group", "full"]

    df["comparison"] = pd.Categorical(
        df["comparison"],
        categories=comparison_order,
        ordered=True
    )

    df["base_channel"] = pd.Categorical(
        df["base_channel"],
        categories=desired_channel_order,
        ordered=True
    )

    # -------------------------------
    # Clean data AFTER categoricals
    # -------------------------------
    df = df.dropna(subset=["value_norm", "base_channel", "comparison"])

    palette = {
        "raw": "#4C72B0",
        "group": "#DD8452",
        "full": "#55A868"
    }

    output_dir = Path(normalized_long_csv).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # -------------------------------
    # Box + strip plot
    # -------------------------------
    sns.boxplot(
        data=df,
        x="base_channel",
        y=y_col,
        hue="comparison",
        hue_order=comparison_order,
        palette=palette,
        dodge=True,
        showfliers=False,
        ax=ax
    )

    sns.stripplot(
        data=df,
        x="base_channel",
        y=y_col,
        hue="comparison",
        hue_order=comparison_order,
        dodge=True,
        palette=palette,
        alpha=0.6,
        size=3,
        jitter=True,
        ax=ax
    )

    # Fix duplicated legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:3],
        labels[:3],
        title="Comparison",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    # -------------------------------
    # Statistics: ANOVA + Tukey
    # -------------------------------
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_step = y_range * 0.04   # vertical spacing between brackets

    for i, channel in enumerate(desired_channel_order):
        sub = df[df["base_channel"] == channel]
        if sub.empty:
            continue

        groups = [
            sub[sub["comparison"] == c][y_col].values
            for c in comparison_order
        ]

        if any(len(g) < 2 for g in groups):
            continue

        # One-way ANOVA
        f_stat, p_anova = f_oneway(*groups)
        if p_anova >= 0.05:
            continue

        # Tukey HSD
        tukey = pairwise_tukeyhsd(
            endog=sub[y_col],
            groups=sub["comparison"],
            alpha=0.05
        )

        tukey_df = pd.DataFrame(
            tukey.summary().data[1:],
            columns=tukey.summary().data[0]
        )

        # 🔑 LOCAL max for this channel
        local_max = sub[y_col].max()
        h = local_max + y_step

        for _, row in tukey_df.iterrows():
            if not row["reject"]:
                continue

            star = p_to_stars(row["p-adj"])
            if star is None:
                continue

            x1 = comparison_order.index(row["group1"])
            x2 = comparison_order.index(row["group2"])

            x_center = i
            offset = [-0.25, 0, 0.25]
            xa = x_center + offset[x1]
            xb = x_center + offset[x2]

            ax.plot([xa, xa, xb, xb], [h, h + y_step, h + y_step, h],
                    lw=1, c="black")
            ax.text((xa + xb) / 2, h + y_step,
                    star, ha="center", va="bottom")

            h += y_step * 1.3

    # Annotate significant comparisons
    local_max = sub[y_col].max()      # max for THIS channel only
    h = local_max + y_step            # start just above the boxes

    for _, row in tukey_df.iterrows():
        if not row["reject"]:
            continue

        star = p_to_stars(row["p-adj"])
        if star is None:
            continue

        x1 = comparison_order.index(row["group1"])
        x2 = comparison_order.index(row["group2"])

        # Convert to axis coordinates
        x_center = i
        offset = [-0.25, 0, 0.25]
        xa = x_center + offset[x1]
        xb = x_center + offset[x2]

        ax.plot(
            [xa, xa, xb, xb],
            [h, h + y_step, h + y_step, h],
            lw=1,
            c="black"
        )
        ax.text(
            (xa + xb) / 2,
            h + y_step,
            star,
            ha="center",
            va="bottom"
        )

        h += y_step * 1.3

    ax.set_xlabel("Channel")
    ax.set_ylabel(y_label)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = output_dir / output_name
    plt.savefig(out, dpi=300, format='svg')
    plt.close()

    print(f"Saved plot with statistics to {out}")

