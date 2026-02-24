"""
Reusable EDA plotting functions for the Spotify bot vs. human analysis.
All functions produce publication-quality figures with labeled axes,
titles, and legends.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Consistent style across all plots
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = ["#1DB954", "#B3B3B3"]  # Spotify green + neutral gray


def descriptive_stats(df, columns=None):
    """
    Return a summary table with count, mean, std, min, quartiles, max
    for the specified columns (or all numeric columns).
    """
    if columns is not None:
        df = df[columns]
    return df.describe().T.round(3)


def plot_histograms(df, columns, bins=25, figsize=None):
    """Plot histograms for a list of continuous columns."""
    n = len(columns)
    if figsize is None:
        figsize = (5 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        ax.hist(df[col].dropna(), bins=bins, color=PALETTE[0],
                edgecolor="white", alpha=0.85)
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {col.replace('_', ' ').title()}")
    plt.tight_layout()
    return fig


def plot_boxplots(df, continuous_col, group_col, figsize=(7, 5)):
    """Boxplot of a continuous variable grouped by a categorical variable."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x=group_col,
        y=continuous_col,
        palette=PALETTE,
        ax=ax,
    )
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel(continuous_col.replace("_", " ").title())
    ax.set_title(
        f"{continuous_col.replace('_', ' ').title()} "
        f"by {group_col.replace('_', ' ').title()}"
    )
    plt.tight_layout()
    return fig


def plot_scatterplot(df, x_col, y_col, hue_col=None, figsize=(7, 5)):
    """Scatterplot with optional hue grouping."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=PALETTE if hue_col else None,
        alpha=0.6,
        ax=ax,
    )
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    title = (
        f"{y_col.replace('_', ' ').title()} vs "
        f"{x_col.replace('_', ' ').title()}"
    )
    ax.set_title(title)
    if hue_col:
        ax.legend(title=hue_col.replace("_", " ").title())
    plt.tight_layout()
    return fig


def plot_pairplot(df, columns, hue_col=None):
    """Seaborn pairplot for a subset of columns."""
    subset = columns.copy()
    if hue_col and hue_col not in subset:
        subset.append(hue_col)
    g = sns.pairplot(
        df[subset].dropna(),
        hue=hue_col,
        palette=PALETTE if hue_col else None,
        diag_kind="hist",
        plot_kws={"alpha": 0.5},
    )
    g.figure.suptitle("Pairplot of Key Variables", y=1.02)
    return g.figure


def plot_correlation_matrix(df, columns, figsize=(8, 6)):
    """Heatmap of the correlation matrix for selected columns."""
    fig, ax = plt.subplots(figsize=figsize)
    corr = df[columns].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        square=True,
    )
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return fig


def plot_class_balance(df, target_col, figsize=(6, 4)):
    """Bar chart showing class distribution for a binary target."""
    fig, ax = plt.subplots(figsize=figsize)
    counts = df[target_col].value_counts().sort_index()
    counts.plot(kind="bar", color=PALETTE, edgecolor="white", ax=ax)
    ax.set_xlabel(target_col.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.set_title(f"Class Distribution: {target_col.replace('_', ' ').title()}")
    ax.set_xticklabels(["Human (0)", "Bot-like (1)"], rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    return fig
