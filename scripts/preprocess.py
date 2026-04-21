import os

import numpy as np
import pandas as pd

ds_name = snakemake.wildcards["preprocess_ds_name"]
out_file = snakemake.output["out_file"]
sample_id = snakemake.config["sample_id"]
method = snakemake.params["method"]
df_mod = snakemake.params["df_mod"]
in_file = snakemake.input["in_file"]

out_dir = os.path.dirname(out_file)
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(in_file, sep="\t")

if df_mod:
    local_vars = {"df": df, "pd": pd, "np": np}
    exec(df_mod, local_vars)
    df = local_vars["df"]

if method in ("clr", "arcsin"):
    s = df[[sample_id]].copy()
    df = df.drop(columns=[sample_id])

    if method == "clr":
        log_df = np.log(df)
        df = log_df.subtract(log_df.mean(axis=1), axis=0)
    elif method == "arcsin":
        df = df.apply(lambda x: np.arcsin(np.sqrt(x)))

    df[sample_id] = s[sample_id].values

df.to_csv(out_file, sep="\t", index=False)
