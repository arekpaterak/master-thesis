from __future__ import annotations

import polars as pl

# One row per problem setting, across every kind. Kind-specific columns are null for
# the other kind (synthetic columns null on CVRPLIB rows and vice versa).
SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    # common
    "config": pl.Utf8,
    "method_type": pl.Utf8,
    "problem_label": pl.Utf8,
    "problem_kind": pl.Utf8,
    "n_instances": pl.Int32,
    "total_time_s": pl.Float64,
    # synthetic
    "problem": pl.Utf8,
    "size": pl.Int32,
    "n_steps": pl.Int32,
    "time_per_instance_s": pl.Float64,
    "obj_initial_mean": pl.Float64,
    "obj_initial_std": pl.Float64,
    "obj_final_mean": pl.Float64,
    "obj_final_std": pl.Float64,
    "obj_final_median": pl.Float64,
    "obj_final_min": pl.Float64,
    "obj_final_max": pl.Float64,
    "improvement_mean": pl.Float64,
    "improvement_std": pl.Float64,
    "improvement_pct_mean": pl.Float64,
    # cvrplib
    "set_name": pl.Utf8,
    "gap_pct_mean": pl.Float64,
    "gap_pct_median": pl.Float64,
    "gap_pct_std": pl.Float64,
    "gap_pct_min": pl.Float64,
    "gap_pct_max": pl.Float64,
    "n_at_or_below_bks": pl.Int32,
}

SYNTHETIC_INSTANCE_SCHEMA: dict[str, pl.DataType] = {
    "config": pl.Utf8,
    "problem_label": pl.Utf8,
    "instance_id": pl.Utf8,
    "seed": pl.Int32,
    "initial_objective": pl.Float64,
    "final_objective": pl.Float64,
    "improvement": pl.Float64,
}

SYNTHETIC_CURVE_SCHEMA: dict[str, pl.DataType] = {
    "lns_step": pl.Int32,
    "obj_mean": pl.Float64,
    "obj_std": pl.Float64,
}

CVRPLIB_INSTANCE_SCHEMA: dict[str, pl.DataType] = {
    "instance": pl.Utf8,
    "n_customers": pl.Int32,
    "bks": pl.Int32,
    "our_cost": pl.Int32,
    "gap_pct": pl.Float64,
}

CVRPLIB_BUCKET_SCHEMA: dict[str, pl.DataType] = {
    "size_bucket": pl.Int32,
    "mean_gap_pct": pl.Float64,
    "median_gap_pct": pl.Float64,
    "count": pl.Int32,
}
