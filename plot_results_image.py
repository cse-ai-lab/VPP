import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_model_name(filename, f='_image_accuracy.csv'):
    """
    Extract model name from filename prefix.
    e.g., 'test_results_cot_gemini_2_5_image_accuracy.csv' → 'cot_gemini_2_5'
    """
    base = os.path.basename(filename)
    # print('base', base)
    model_name = base.replace('test_results_', '')
    # print('model_name', model_name)
    model_name = model_name.replace(f, '')
    # print('model_name', model_name)
    return model_name

def load_accuracy_files(filepaths, f):
    """
    Load multiple image_accuracy CSVs and annotate with model name.
    Returns a combined DataFrame.
    """
    dfs = []
    for path in filepaths:
        model_name = extract_model_name(path, f)
        df = pd.read_csv(path)
        df['model'] = model_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# def plot_image_accuracy_distribution(df, output_path="./new/image_accuracy_comparison.jpg"):
#     """
#     Generate and save a CVPR-style professional plot comparing image-level accuracy across models.
#     """
#     # Compute mean accuracy per model for sorting
#     model_order = df.groupby("model")["accuracy"].mean().sort_values(ascending=False).index.tolist()

#     # Set style and font
#     sns.set_theme(context="paper", style="whitegrid", font="serif", font_scale=1.5)

#     plt.figure(figsize=(10, 6))

#     # Boxplot with mean markers
#     ax = sns.boxplot(
#         data=df,
#         x="model",
#         y="accuracy",
#         order=model_order,
#         palette="colorblind",
#         showmeans=True,
#         meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
#         flierprops={"marker": "o", "markersize": 4, "alpha": 0.4}
#     )

#     # Annotate mean accuracy above each box
#     means = df.groupby("model")["accuracy"].mean().reindex(model_order)
#     for i, (model, mean_acc) in enumerate(means.items()):
#         ax.text(i, 1.02, f"{mean_acc:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

#     # Axis labels and title
#     plt.title("Image-level Accuracy Distribution by Model", fontsize=16, weight='bold')
#     plt.xlabel("Model", fontsize=14)
#     plt.ylabel("Accuracy per Image", fontsize=14)
#     plt.ylim(0, 1.1)
#     plt.grid(True, axis='y', linestyle='--', alpha=0.5)

#     # Tight layout and export
#     plt.tight_layout()
#     plt.savefig(output_path)
#     print(f"✅ Saved high-resolution plot to: {output_path}")
#     plt.close()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# def plot_per_image_accuracy(df, output_path="./new/accuracy_per_image_lineplotjpg"):
#     """
#     Plots a line plot of per-image accuracy across models.
#     Expects DataFrame with columns: image, accuracy, model
#     """
#     # Sort image names for consistent ordering
#     df_sorted = df.copy()
#     df_sorted["image_idx"] = df_sorted["image"].astype("category").cat.codes
#     df_sorted = df_sorted.sort_values(by=["image_idx", "model"])

#     # Pivot for wide format
#     df_pivot = df_sorted.pivot(index="image", columns="model", values="accuracy").reset_index()

#     # Sort by average image difficulty
#     df_pivot["mean_accuracy"] = df_pivot.drop(columns=["image"]).mean(axis=1)
#     df_pivot = df_pivot.sort_values(by="mean_accuracy", ascending=False).drop(columns=["mean_accuracy"])

#     # Re-pivot long for Seaborn
#     df_long = df_pivot.melt(id_vars="image", var_name="model", value_name="accuracy")
#     df_long["image_idx"] = df_long.groupby("image").ngroup()

#     # Plot
#     sns.set(style="whitegrid", font_scale=1.5)
#     plt.figure(figsize=(16, 6))
#     sns.lineplot(data=df_long, x="image_idx", y="accuracy", hue="model", marker="o", linewidth=2)

#     plt.title("Image-wise Accuracy Across Models", fontsize=16, weight='bold')
#     plt.xlabel("Images (Sorted by Difficulty)", fontsize=14)
#     plt.ylabel("Accuracy", fontsize=14)
#     plt.ylim(0, 1.05)
#     plt.legend(title="Model", loc="lower right")
#     plt.tight_layout()

#     plt.savefig(output_path, dpi=600)
#     plt.close()
#     print(f"✅ Saved line plot to {output_path}")




# def plot_violin_accuracy_distribution(df, output_path="./new/violin_accuracy_distribution.pdf"):
#     """
#     Generate a violin plot showing image-level accuracy distributions per model.
#     """
#     sns.set_theme(style="whitegrid", font="serif", font_scale=1.4)
#     plt.figure(figsize=(10, 6))

#     ax = sns.violinplot(
#         data=df,
#         x="model",
#         y="accuracy",
#         palette="colorblind",
#         cut=0,
#         inner="quartile",  # shows median + IQR
#         linewidth=1.2
#     )

#     # Add mean as black dots
#     means = df.groupby("model")["accuracy"].mean()
#     for i, (model, mean_val) in enumerate(means.items()):
#         ax.scatter(i, mean_val, color="black", marker="o", s=60, zorder=5)
#         ax.text(i, 1.03, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

#     # Title and labels
#     plt.title("Violin Plot of Image-Level Accuracy by Model", fontsize=16, weight='bold')
#     plt.xlabel("Model", fontsize=14)
#     plt.ylabel("Accuracy per Image", fontsize=14)
#     plt.ylim(0, 1.1)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=600)
#     plt.close()
#     print(f"✅ Saved violin plot to {output_path}")



def plot_violin_accuracy_distribution(df, output_path):
    """
    Generate a violin plot showing image-level accuracy distributions per model,
    with annotated mean and correct/total counts.
    """
    sns.set_theme(style="whitegrid", font="serif", font_scale=1.4)
    plt.figure(figsize=(10, 6))

    ax = sns.violinplot(
        data=df,
        x="model",
        y="accuracy",
        palette="colorblind",
        cut=0,
        inner="quartile",  # median + IQR lines
        linewidth=1.2
    )
    # print('*****'*20)
    # print(df)
    # Compute stats per model
    stats = df.groupby(["model"]).agg(
        mean_accuracy=("accuracy", "mean"),
        num_correct=("num_correct", "sum"),
        total=("total", "sum")
    ).reset_index()
    # print(stats.head())

    # Add annotations above each violin
    for i, row in stats.iterrows():
        mean_val = row['mean_accuracy']
        correct = int(row['num_correct'])
        total = int(row['total'])

        ax.scatter(i, mean_val, color="black", marker="o", s=60, zorder=5)
        ax.text(i, 1.03,
                f"Mean: {mean_val:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold", linespacing=1.4)

    # Titles and axes
    # plt.title("Violin Plot of Image-Level Accuracy by Model", fontsize=16, weight='bold')
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Accuracy per AST", fontsize=14)
    plt.ylim(0, 1.12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"✅ Saved violin plot with stats to {output_path}")



if __name__ == "__main__":
    pref = ['test_results_cot_gemini_2_5', 'test_results_cot_gpt4o', 'test_results_gemini_2_5', 'test_results_gpt4o' ]
    post ='_qid_accuracy.csv'
    csv_files = [f'{_}{post}' for _ in pref]
    df_all = load_accuracy_files(csv_files, '_qid_accuracy.csv')
    plot_violin_accuracy_distribution(df_all, './new/accuracy_qid_distribution.jpg')

    post ='_image_accuracy.csv'
    csv_files = [f'{_}{post}' for _ in pref]
    df_all = load_accuracy_files(csv_files, '_image_accuracy.csv')
    plot_violin_accuracy_distribution(df_all, './new/accuracy_image_distribution.jpg')

    post ='_tag_accuracy.csv'
    csv_files = [f'{_}{post}' for _ in pref]
    df_all = load_accuracy_files(csv_files, '_tag_accuracy.csv')
    plot_violin_accuracy_distribution(df_all, './new/accuracy_tag_distribution.jpg')

    post ='_source_accuracy.csv'
    csv_files = [f'{_}{post}' for _ in pref]
    df_all = load_accuracy_files(csv_files, '_source_accuracy.csv')
    plot_violin_accuracy_distribution(df_all, './new/accuracy_source_distribution.jpg')

    '''
    # print(df_all)
    # plot_image_accuracy_distribution(df_all)
    # plot_per_image_accuracy(df_all)
    '''
