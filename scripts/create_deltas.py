import os

import numpy as np
import pandas as pd

output_folder = snakemake.config["out"]
response_var = snakemake.config["response_var"]
sample_id = snakemake.config["sample_id"]
input_study_id = snakemake.config["random_effect"]
analyze_first = snakemake.config["analyze_first"]
analyze_previous = snakemake.config["analyze_previous"]
analyze_pairwise = snakemake.config["analyze_pairwise"]
include_reference_values = snakemake.params["include_reference_values"]
absolute_values = snakemake.params["absolute_values"]
in_file = snakemake.input["in_file"]
out_file = snakemake.output["out_file"]
reference_time = snakemake.params["reference_time"]
distance_matrices_config = snakemake.params["distance_matrices"]

input_timepoint = "timepoint_explana"

# distance_matrices_config is a dict {name: path} or empty
if not distance_matrices_config:
    distance_matrices_config = {}


def load_distance_matrices(dm_config):
    dms = {}
    for name, path in dm_config.items():
        dm = pd.read_csv(path, sep="\t", index_col=0)
        dms[name] = dm
    return dms


def write_failed(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("Failed Analysis")


def should_skip():
    if analyze_first != "yes" and reference_time == "first":
        return True
    if analyze_previous != "yes" and reference_time == "previous":
        return True
    if analyze_pairwise != "yes" and reference_time == "pairwise":
        return True
    return False


def main():
    if should_skip():
        write_failed(out_file)
        return

    df = pd.read_csv(in_file, sep="\t")

    total_timepoints = df[input_timepoint].nunique()

    if total_timepoints == 1 and reference_time == "first":
        write_failed(out_file)
        return

    if total_timepoints <= 2 and reference_time in ("previous", "pairwise"):
        write_failed(out_file)
        return

    distance_matrices = load_distance_matrices(distance_matrices_config)

    df = df.rename(columns={
        input_timepoint: "timepoint_w_",
        input_study_id: "study_id_w_",
    })

    df[sample_id] = df[sample_id].astype(str)

    vars_list = [c for c in df.columns if c != "study_id_w_"]
    times = sorted(df["timepoint_w_"].unique())

    delta_rows = []

    for studyid in df["study_id_w_"].unique():
        for time in times:
            time = int(time)

            if reference_time == "first":
                ref_times = [1]
            elif reference_time == "previous":
                ref_times = [time - 1]
            else:  # pairwise
                ref_times = [int(t) for t in times if int(t) != time]

            for rt in ref_times:
                if rt >= time:
                    continue

                mask = (
                    df["study_id_w_"].eq(studyid) &
                    df["timepoint_w_"].isin([rt, time])
                )
                sid_df = df[mask]

                if len(sid_df) < 2:
                    continue

                ref_row = sid_df[sid_df["timepoint_w_"] == rt]
                cur_row = sid_df[sid_df["timepoint_w_"] == time]

                if ref_row.empty or cur_row.empty:
                    continue

                ref_sample = ref_row.iloc[0]
                cur_sample = cur_row.iloc[0]

                new_row = {
                    "sid_delta": f"{studyid}__{rt}__{time}",
                    "timepoint_w_": f"{rt}__{time}",
                    "study_id_w_": studyid,
                }

                for dm_name, dm in distance_matrices.items():
                    try:
                        ref_sid = ref_sample[sample_id]
                        cur_sid = cur_sample[sample_id]
                        dist = dm.loc[ref_sid, cur_sid]
                        new_row[dm_name] = dist
                    except Exception:
                        new_row[dm_name] = np.nan

                for var in vars_list:
                    new_val = cur_sample[var]
                    old_val = ref_sample[var]

                    is_numeric = pd.api.types.is_numeric_dtype(df[var])
                    is_special = var in ("timepoint_w_", sample_id)

                    if is_numeric and not is_special:
                        if absolute_values == "yes":
                            new_row[var] = abs(new_val - old_val)
                        else:
                            new_row[var] = new_val - old_val
                        new_row[f"{var}_reference"] = old_val

                    if not is_numeric or is_special:
                        new_row[var] = f"{old_val}__{new_val}"
                        new_row[f"{var}_reference"] = str(old_val)

                delta_rows.append(new_row)

    if not delta_rows:
        write_failed(out_file)
        return

    delta_df = pd.DataFrame(delta_rows)
    delta_df = delta_df.sort_values(["study_id_w_", "timepoint_w_"])
    delta_df = delta_df.drop(columns=["sid_delta"])

    delta_df = delta_df.rename(columns={
        "timepoint_w_": input_timepoint,
        "study_id_w_": input_study_id,
    })

    if include_reference_values == "no":
        ref_cols = [c for c in delta_df.columns if c.endswith("_reference")]
        delta_df = delta_df.drop(columns=ref_cols)

    delta_df_mod = snakemake.config.get("delta_df_mod", "")
    if delta_df_mod:
        local_vars = {"delta_df": delta_df, "pd": pd, "np": np}
        exec(delta_df_mod, local_vars)
        delta_df = local_vars["delta_df"]

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    delta_df.to_csv(out_file, sep="\t", index=False)


main()
