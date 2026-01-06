import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_model_name(filename, f='_source_accuracy.csv'):
    """
    Extract model name from filename prefix.
    e.g., 'test_results_cot_gemini_2_5_tag_accuracy.csv' → 'cot_gemini_2_5'
    """
    base = os.path.basename(filename)
    model_name = base.replace('test_results_', '').replace(f, '')
    return model_name

def load_accuracy_files(filepaths):
    """
    Load multiple tag_accuracy CSVs and annotate with model name.
    Returns a combined DataFrame.
    """
    dfs = []
    for path in filepaths:
        model_name = extract_model_name(path)
        df = pd.read_csv(path)
        df['model'] = model_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def plot_bar_tag(df_):
    """
    Generate a clean, publication-quality horizontal bar plot of accuracy per tag across models.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Pivot the data: tag × model → accuracy
    pivot_df = df_.pivot(index="source", columns="model", values="accuracy").fillna(0)

    # Sort by mean accuracy (descending) for readability
    pivot_df = pivot_df.loc[pivot_df.mean(axis=1).sort_values(ascending=True).index]

    # Set seaborn style and subdued color palette
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", n_colors=len(pivot_df.columns))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot_df) * 0.25)))

    # Plot grouped bars
    pivot_df.plot(
        kind="barh",
        ax=ax,
        color=palette,
        width=0.75,
        edgecolor="none"
    )

    # Axis labels and title
    ax.set_xlabel("Accuracy", fontsize=13, labelpad=8)
    ax.set_ylabel("Premise Type", fontsize=13, labelpad=8)
    ax.set_title("Accuracy per Premise Type Across Models", fontsize=14, pad=12, weight="semibold")

    # Ticks and formatting
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlim(0, 1.0)

    # Add numerical annotations for each bar
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            label_type="edge",
            padding=1,
            fontsize=8,
            color="dimgray"
        )

    # Legend formatting
    ax.legend(
        title="Model",
        loc="best",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    # Grid and layout
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    # Save high-quality JPG
    output_path = '/Users/s0a0igg/Documents/reps/VPP/new/accuracy_bar_source.jpg'
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Plot saved to: {output_path}")

def plot_bar_tag_(df_):
    """
    Generate a clean, publication-quality stacked vertical bar plot of accuracy per tag across models.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Pivot: tag as x-axis, models as stacked segments
    pivot_df = df_.pivot(index="source", columns="model", values="accuracy").fillna(0)

    # Sort tags by mean accuracy (optional for better flow)
    pivot_df = pivot_df.loc[pivot_df.mean(axis=1).sort_values(ascending=False).index]

    # Set seaborn style and subdued color palette
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set2", n_colors=pivot_df.columns.size)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, len(pivot_df) * 0.35), 6))

    # Plot stacked bars manually
    bottoms = pd.Series([0] * len(pivot_df), index=pivot_df.index)
    for i, model in enumerate(pivot_df.columns):
        ax.bar(
            pivot_df.index,
            pivot_df[model],
            bottom=bottoms,
            label=model,
            color=palette[i],
            edgecolor='none',
            width=0.8
        )
        bottoms += pivot_df[model]

    # Labels and title
    ax.set_ylabel("Accuracy", fontsize=13, labelpad=8)
    ax.set_xlabel("Premise Tag", fontsize=13, labelpad=8)
    ax.set_title("Stacked Accuracy per Premise Tag Across Models", fontsize=14, pad=12, weight="semibold")

    # Axis formatting
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.index, rotation=90, fontsize=8)
    ax.tick_params(axis='y', labelsize=10)

    # Legend
    ax.legend(
        title="Model",
        loc="upper right",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    # Grid and cleanup
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    # Save high-quality JPG
    output_path = '/Users/s0a0igg/Documents/reps/VPP/new/accuracy_bar_source_vertical.jpg'
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Plot saved to: {output_path}")

def plot_bar_tag_grid(df_):
    """
    Create a 2x2 grid of vertical bar plots (one per model) showing per-tag accuracy,
    each with a unique color and a median reference line.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Config
    models = df_["model"].unique()
    n_models = len(models)
    assert n_models == 4, f"Expected 4 models, got {n_models}"

    tags = sorted(df_["qid"].unique())

    # Set style
    sns.set_theme(style="whitegrid")
    model_colors = sns.color_palette("Set2", n_colors=n_models)

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        sub_df = df_[df_["model"] == model].copy()
        sub_df = sub_df.set_index("qid").reindex(tags).fillna(0)

        # Plot vertical bar chart
        ax.bar(
            sub_df.index,
            sub_df["accuracy"],
            color=model_colors[i],
            edgecolor="none",
            width=0.75
        )

        # Median reference line
        median_val = sub_df["accuracy"].median()
        ax.axhline(median_val, color='dimgray', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(
            len(sub_df) - 1, median_val + 0.015,
            f"Median = {median_val:.2f}",
            color="dimgray", fontsize=8, ha='right'
        )

        # Formatting
        ax.set_title(f"{model}", fontsize=12, weight='semibold')
        ax.set_xticks(range(len(sub_df)))
        ax.set_xticklabels(sub_df.index, rotation=90, fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.tick_params(axis='y', labelsize=9)

    # Global layout
    fig.suptitle("Per-AST Accuracy per Model (Median Line Included)", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    sns.despine()

    # Save output
    output_path = "/Users/s0a0igg/Documents/reps/VPP/new/accuracy_per_qid_grid.jpg"
    plt.savefig(output_path, dpi=800, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot saved to: {output_path}")


if __name__ == "__main__":
    csv_files = [
        '/Users/s0a0igg/Documents/reps/VPP/test_results_cot_gemini_2_5_source_accuracy.csv', 
        '/Users/s0a0igg/Documents/reps/VPP/test_results_gemini_2_5_source_accuracy.csv',
    '/Users/s0a0igg/Documents/reps/VPP/test_results_cot_gpt4o_source_accuracy.csv', 
        '/Users/s0a0igg/Documents/reps/VPP/test_results_gpt4o_source_accuracy.csv',
    ]
    df_all = load_accuracy_files(csv_files)
    print(df_all)
    plot_bar_tag(df_all)
    # plot_bar_tag_grid(df_all)
