import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

selected_features_original = snakemake.input["selected_features_original"]
selected_features_first = snakemake.input["selected_features_first"]
selected_features_previous = snakemake.input["selected_features_previous"]
selected_features_pairwise = snakemake.input["selected_features_pairwise"]

base_path = snakemake.config["out"]
out_file = snakemake.output["out_file"]
out_file_image = snakemake.output["out_file_image"]

COLORS = {
    "Original": "#9EBCDA",
    "First": "#CBC9E2",
    "Previous": "#9E9AC8",
    "Pairwise": "#756BB1",
}
TRUNC_LEN = 40


def truncate_label(label, max_len=TRUNC_LEN):
    if len(label) <= max_len:
        return label
    half = max_len // 2
    return label[:half] + "..." + label[-(half - 1):]


def load_boruta_file(path, ds_label):
    df = pd.read_csv(path, sep="\t")
    if "Failed Analysis" in df.columns:
        return pd.DataFrame({
            "important_features": ["Failed Analysis"],
            "decoded_features": ["Failed Analysis"],
            "feature_importance_vals": [10000],
        })
    return df


def import_ds_for_summary(path, ds_label):
    df = load_boruta_file(path, ds_label)
    max_rank = len(df["feature_importance_vals"])
    df["ranked_importance"] = (
        max_rank + 1 - df["feature_importance_vals"].rank(method="first").astype(int)
    )
    df[ds_label] = 1
    df["dataset"] = ds_label
    return df


def add_average_shap_values(df, ds):
    ds_lower = ds.lower()
    type_path = f"path_{ds_lower}"
    path_prefix = base_path + snakemake.config[type_path] + ds_lower

    input_fp = path_prefix + "-input-model-df.txt"
    shap_fp = path_prefix + "-SHAP-values-df.txt"

    input_features = []
    if not os.path.exists(input_fp):
        return df, input_features

    df_input = pd.read_csv(input_fp, sep="\t")
    input_features = df_input.columns.tolist()

    if not os.path.exists(shap_fp):
        return df, input_features

    df_shap = pd.read_csv(shap_fp, sep="\t")

    enc_features = df[
        (df["dataset"] == ds) & df["important_features"].str.startswith("ENC_", na=False)
    ]["important_features"]

    for feat in enc_features:
        if feat not in df_input.columns or feat not in df_shap.columns:
            continue
        mask = df_input[feat] == 1
        avg_shap = round(df_shap.loc[mask, feat].mean(), 3)
        cond = (df["important_features"] == feat) & (df["dataset"] == ds)
        df.loc[cond, "shap_val"] = avg_shap

    return df, input_features


def make_summary_log_dataframe():
    order = [
        "Model Type (Evaluation)", "% Variance Explained", "N Trees",
        "Feature fraction/split", "Max Depth", "MERF Iters.",
        "BorutaSHAP Trials", "BorutaSHAP Threshold", "P-value",
        "N Study IDs", "N Samples", "Input Features",
        "Accepted Features", "Tentative Features", "Rejected Features",
    ]
    frames = []
    for ds in ("Original", "First", "Previous", "Pairwise"):
        type_path = f"path_{ds.lower()}"
        fp = base_path + snakemake.config[type_path] + ds.lower() + "-log-df.txt"
        if os.path.exists(fp):
            frames.append(pd.read_csv(fp, sep="\t"))

    if frames:
        df = pd.concat(frames, ignore_index=True)
        existing_order = [o for o in order if o in df.columns]
        df = df[existing_order] if existing_order else df
        df.to_csv(base_path + "summary-log-table.txt", sep="\t", index=False)


def main():
    make_summary_log_dataframe()

    df_original = import_ds_for_summary(selected_features_original, "Original")
    df_first = import_ds_for_summary(selected_features_first, "First")
    df_previous = import_ds_for_summary(selected_features_previous, "Previous")
    df_pairwise = import_ds_for_summary(selected_features_pairwise, "Pairwise")

    df_join = pd.concat(
        [df_original, df_first, df_previous, df_pairwise],
        ignore_index=True, sort=False
    )
    df_join = df_join.fillna(0)

    df_join["shap_val"] = 0.0

    df_join, input_features_original = add_average_shap_values(df_join, "Original")
    df_join, input_features_first = add_average_shap_values(df_join, "First")
    df_join, input_features_previous = add_average_shap_values(df_join, "Previous")
    df_join, input_features_pairwise = add_average_shap_values(df_join, "Pairwise")

    all_input_features = set(
        input_features_original + input_features_first +
        input_features_previous + input_features_pairwise
    )

    # Write all-important-features.txt
    df_all = df_join[["dataset", "important_features"]].copy()
    df_all = df_all[df_all["important_features"] != "Failed Analysis"]
    df_all.to_csv(out_file, sep="\t", index=False)

    # Build long-form dataframe for the occurrence plot
    datasets = ["Original", "First", "Previous", "Pairwise"]
    features = df_join["important_features"].unique().tolist()

    rows = []
    for feat in features:
        feat_rows = df_join[df_join["important_features"] == feat]
        decoded = feat_rows["decoded_features"].iloc[0] if not feat_rows.empty else feat
        for ds in datasets:
            ds_feat = feat_rows[feat_rows["dataset"] == ds]
            selected = 1 if not ds_feat.empty else 0
            ranked = int(ds_feat["ranked_importance"].iloc[0]) if selected else 0
            shap_val = float(ds_feat["shap_val"].iloc[0]) if selected else 0.0
            is_model_input = (
                feat in all_input_features and not selected
            )
            if selected and shap_val != 0.0:
                fig_label = f"{ranked}: {shap_val}"
            elif selected:
                fig_label = str(ranked)
            else:
                fig_label = ""

            rows.append({
                "important_features": feat,
                "decoded_features": decoded,
                "dataset": ds,
                "value": selected,
                "ranked_importance": ranked,
                "shap_val": shap_val,
                "figure_labels": fig_label,
                "is_model_input": " " if is_model_input else "",
            })

    df_long = pd.DataFrame(rows)

    num_failed = len(df_long[df_long["important_features"] == "no_selected_features"]["dataset"].unique())
    if num_failed != 4:
        df_long = df_long[df_long["important_features"] != "no_selected_features"]

    df_long.to_csv(base_path + "df_join_long.txt", sep="\t", index=False)

    _make_occurrence_plot(df_long, out_file_image)


def _make_occurrence_plot(df_long, svg_path):
    features = df_long["important_features"].unique().tolist()
    # Filter to only selected features for layout, preserve order by rank
    selected_mask = df_long["value"] == 1
    feat_order = (
        df_long[selected_mask]
        .groupby("important_features")["ranked_importance"]
        .min()
        .sort_values()
        .index.tolist()
    )
    # append any remaining features not selected in any model
    remaining = [f for f in features if f not in feat_order]
    feat_order = feat_order + remaining

    datasets = ["Original", "First", "Previous", "Pairwise"]

    n_feats = len(feat_order)
    n_ds = len(datasets)

    height_ratio = 12 / 40 if n_feats < 20 else 10.8 / 40
    height = max(4.0, n_feats * height_ratio)

    fig, ax = plt.subplots(figsize=(9, height))

    for row_idx, feat in enumerate(feat_order):
        feat_df = df_long[df_long["important_features"] == feat]
        y = n_feats - row_idx - 1  # top to bottom

        for col_idx, ds in enumerate(datasets):
            ds_row = feat_df[feat_df["dataset"] == ds]
            selected = bool(ds_row["value"].iloc[0]) if not ds_row.empty else False
            label = ds_row["figure_labels"].iloc[0] if not ds_row.empty else ""
            is_input = ds_row["is_model_input"].iloc[0] if not ds_row.empty else ""

            facecolor = COLORS[ds] if selected else "white"
            alpha = 1.0 if selected else 0.0

            rect = mpatches.FancyBboxPatch(
                (col_idx + 0.05, y + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=facecolor, edgecolor="#bfbfbf",
                alpha=1.0 if selected else 0.15,
                linewidth=0.7,
            )
            ax.add_patch(rect)

            if selected and label:
                ax.text(
                    col_idx + 0.5, y + 0.5, label,
                    ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold",
                )
            elif is_input == " ":
                ax.text(
                    col_idx + 0.5, y + 0.5, "·",
                    ha="center", va="center",
                    fontsize=8, color="#9e9c9c",
                )

    def get_n(ds):
        n = df_long[(df_long["dataset"] == ds) & (df_long["value"] == 1)]["ranked_importance"].max()
        return f"\n   N:{int(n)}" if pd.notna(n) and n > 0 else ""

    ax.set_xlim(0, n_ds)
    ax.set_ylim(0, n_feats)
    ax.set_xticks([i + 0.5 for i in range(n_ds)])
    ax.set_xticklabels(
        [f"{ds}{get_n(ds)}" for ds in datasets],
        fontsize=8, fontweight="bold",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0, pad=4)

    y_labels = [truncate_label(f) for f in feat_order]
    ax.set_yticks(np.arange(n_feats) + 0.5)
    ax.set_yticklabels(y_labels[::-1], fontsize=7)
    ax.tick_params(axis="y", length=0)

    ax.set_xlabel("Model Comparison", fontsize=9)
    ax.set_ylabel("Selected Features", fontsize=9)

    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#bfbfbf")

    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color="#f2f2f2", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


main()
