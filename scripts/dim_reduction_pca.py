import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ds_name = snakemake.wildcards["pca_ds_name"]
in_file = snakemake.input["in_file"]
out_file = snakemake.output["out_file"]
sample_id = snakemake.config["sample_id"]
pca_list = snakemake.params["pca_list"]

out_base = snakemake.config["out"] + snakemake.config["path_dim_pca"] + ds_name + "/"

df = pd.read_csv(in_file, sep="\t")


def write_tsv(frame, folder, filename):
    os.makedirs(folder, exist_ok=True)
    frame.to_csv(os.path.join(folder, filename), sep="\t")


def perform_pca(pca_name, pca_config, df):
    feature_list = pca_config["feature_list"]
    if isinstance(feature_list, str):
        # support legacy comma-separated string for convenience
        feature_list = [f.strip().strip("'\"") for f in feature_list.split(",")]
    cum_var_thres = float(pca_config["cum_var_thres"])

    output_dir = os.path.join(out_base, pca_name) + "/"
    os.makedirs(output_dir, exist_ok=True)

    pca_df = df[feature_list]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pca_df)

    pca = PCA()
    ind_coords = pca.fit_transform(scaled)

    explained = pca.explained_variance_ratio_ * 100
    cum_explained = np.cumsum(explained)

    # Select PCs below threshold, plus one more
    below = np.sum(cum_explained < cum_var_thres)
    n_pcs = int(below) + 1
    pc_names = [f"Dim.{i+1}" for i in range(len(explained))]

    # Individual coordinates, contributions, cos2
    ind_coord_df = pd.DataFrame(ind_coords, columns=pc_names)
    total_var = np.sum(pca.explained_variance_)
    ind_contrib_df = (ind_coord_df ** 2).div(
        (ind_coord_df ** 2).sum(axis=0), axis=1
    ) * 100
    ind_cos2_df = (ind_coord_df ** 2).div(
        (ind_coord_df ** 2).sum(axis=1), axis=0
    )

    write_tsv(ind_coord_df, output_dir, "ind_coord.txt")
    write_tsv(ind_contrib_df, output_dir, "ind_contrib.txt")
    write_tsv(ind_cos2_df, output_dir, "ind_cos2.txt")

    # Variable loadings, correlations, contributions, cos2
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    var_coord_df = pd.DataFrame(loadings, index=feature_list, columns=pc_names)
    var_cor_df = pd.DataFrame(
        pca.components_.T, index=feature_list, columns=pc_names
    )
    var_contrib_df = (var_coord_df ** 2).div(
        pca.explained_variance_, axis=1
    ) * 100
    var_cos2_df = var_coord_df ** 2

    write_tsv(var_coord_df, output_dir, "var_coord.txt")
    write_tsv(var_cor_df, output_dir, "var_cor.txt")
    write_tsv(var_contrib_df, output_dir, "var_contrib.txt")
    write_tsv(var_cos2_df, output_dir, "var_cos2.txt")

    # Scree plot — top PCs
    display_n = min(n_pcs, 100)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(range(1, display_n + 1), explained[:display_n])
    for i, v in enumerate(explained[:display_n]):
        ax.text(i + 1, v + 0.2, f"{v:.1f}%", ha="center", fontsize=7)
    ax.set_title("Scaled Scree Plot")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    fig.savefig(os.path.join(output_dir, "scaled-scree-plot-top-pcs.pdf"))
    plt.close(fig)

    # Scree plot — all PCs
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(range(1, len(explained) + 1), explained)
    ax.set_title("Scaled Scree Plot (All PCs)")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    fig.savefig(os.path.join(output_dir, "scaled-scree-plot-all.pdf"))
    plt.close(fig)

    # Variable PCA plot (top 2 dims)
    fig, ax = plt.subplots(figsize=(14, 8))
    contributions = var_contrib_df["Dim.1"].values if "Dim.1" in var_contrib_df else np.zeros(len(feature_list))
    sc = ax.scatter(var_coord_df["Dim.1"], var_coord_df["Dim.2"], c=contributions, cmap="RdYlBu_r")
    for i, feat in enumerate(feature_list):
        ax.annotate(feat, (var_coord_df["Dim.1"].iloc[i], var_coord_df["Dim.2"].iloc[i]), fontsize=7)
    plt.colorbar(sc, ax=ax, label="Contribution to PC1 (%)")
    ax.set_xlabel("Dim.1")
    ax.set_ylabel("Dim.2")
    ax.set_title("Variables PCA")
    fig.savefig(os.path.join(output_dir, "variables-pca.pdf"))
    plt.close(fig)

    # Add selected PC columns to df and remove original features
    selected_pc_names = pc_names[:n_pcs]
    for i, pc in enumerate(selected_pc_names):
        col_name = f"{pca_name}_{pc}"
        df[col_name] = ind_coords[:, i]
    df = df.drop(columns=feature_list)

    df.to_csv(
        os.path.join(output_dir, "pca-dim-reduction-partial-df.txt"),
        sep="\t", index=False
    )
    return df


os.makedirs(out_base, exist_ok=True)

for pca_name, pca_config in pca_list.items():
    df = perform_pca(pca_name, pca_config, df)

df.to_csv(out_file, sep="\t", index=False)
