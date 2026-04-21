import os

import numpy as np
import pandas as pd

out_file = snakemake.output["out_file"]
sample_id = snakemake.config["sample_id"]
timepoint_config = snakemake.config["timepoint"]
df_mod = snakemake.config.get("df_mod", "")

file_path_list = snakemake.input["file_path_list"]
if isinstance(file_path_list, str):
    file_path_list = [file_path_list]

out_dir = os.path.dirname(out_file)
os.makedirs(out_dir, exist_ok=True)

df_list = [pd.read_csv(fp, sep="\t") for fp in file_path_list]

df_complete = df_list[0]
for df in df_list[1:]:
    shared_cols = set(df_complete.columns) & set(df.columns) - {sample_id}
    if shared_cols:
        print(f"WORKFLOW WARNING: duplicate column names across datasets: {shared_cols}")
    df_complete = df_complete.merge(df, on=sample_id, how="inner")

if df_mod:
    local_vars = {"df": df_complete, "pd": pd, "np": np}
    exec(df_mod, local_vars)
    df_complete = local_vars["df"]

df_complete["timepoint_original"] = df_complete[timepoint_config]
df_complete["timepoint_explana"] = (
    df_complete[timepoint_config].rank(method="dense").astype(int)
)
df_complete = df_complete.sort_values("timepoint_original")
df_complete = df_complete.drop(columns=[timepoint_config])

df_complete.to_csv(out_file, sep="\t", index=False)
