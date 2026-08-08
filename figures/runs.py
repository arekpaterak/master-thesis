from __future__ import annotations

WANDB_PROJECT = "nds"

METHOD_COLORS: dict[str, str] = {
    "Random": "tab:blue",
    "Joint + AR": "tab:orange",
    "Joint + NAR": "tab:green",
    "Split + AR": "tab:purple",
    "Split + NAR": "tab:red",
}

# Distinct marker per method so series stay identifiable in grayscale and where lines
# overlap, shared across figures for consistency with METHOD_COLORS.
METHOD_MARKERS: dict[str, str] = {
    "Random": "o",
    "Joint + AR": "s",
    "Joint + NAR": "^",
    "Split + AR": "D",
    "Split + NAR": "v",
}


def variant_train_runs(
    *, joint_ar: str = "", joint_nar: str = "", split_ar: str = "", split_nar: str = ""
) -> list[tuple[str, str]]:
    return [
        ("Joint + AR", joint_ar),
        ("Joint + NAR", joint_nar),
        ("Split + AR", split_ar),
        ("Split + NAR", split_nar),
    ]


TRAIN_RUNS: dict[str, dict] = {
    "TSP50": {
        "problem": "tsp",
        "task": "tsp50",
        "runs": variant_train_runs(
            # joint_ar=".../featured/...",   # not identified yet
            # joint_nar=".../featured/...",  # not identified yet
            # split_ar=".../featured/...",   # not identified yet
            # split_nar=".../featured/...",  # not identified yet
        ),
    },
    "TSP100": {
        "problem": "tsp",
        "task": "tsp100",
        "runs": variant_train_runs(
            joint_ar="tsp/featured/nds_tsp_100+ar",
            joint_nar="tsp/featured/nds_tsp_100+simple",
            # split_ar="tsp/featured/...",   # no featured run yet
            split_nar="tsp/featured/nds_tsp_100_split+simple",
        ),
    },
    "CVRP50": {
        "problem": "cvrp",
        "task": "cvrp50",
        "runs": variant_train_runs(
            # joint_ar=".../featured/...",   # not identified yet
            # joint_nar=".../featured/...",  # not identified yet
            # split_ar=".../featured/...",   # not identified yet
            # split_nar=".../featured/...",  # not identified yet
        ),
    },
    "CVRP100": {
        "problem": "cvrp",
        "task": "cvrp100",
        "runs": variant_train_runs(
            joint_ar="cvrp/featured/nds_cvrp_100+ar",
            joint_nar="cvrp/featured/nds_cvrp_100+simple",
            # split_ar="cvrp/featured/...",  # no featured run yet
            split_nar="cvrp/featured/nds_cvrp_100_split+simple",
        ),
    },
    "CMSTP50": {
        "problem": "cmstp",
        "task": "cmstp50",
        "runs": variant_train_runs(
            # joint_ar=".../featured/...",   # not identified yet
            # joint_nar=".../featured/...",  # not identified yet
            # split_ar=".../featured/...",   # not identified yet
            # split_nar=".../featured/...",  # not identified yet
        ),
    },
    "CMSTP100": {
        "problem": "cmstp",
        "task": "cmstp100",
        "runs": variant_train_runs(
            # joint_ar=".../featured/...",   # not identified yet
            # joint_nar=".../featured/...",  # not identified yet
            # split_ar=".../featured/...",   # not identified yet
            # split_nar=".../featured/...",  # not identified yet
        ),
    },
}


def variant_eval_runs(
    *,
    joint_ar: str = "",
    joint_nar: str = "",
    split_ar: str = "",
    split_nar: str = "",
    random: str = "",
) -> list[tuple[str, str]]:
    """Ordered ``(method label, eval config name)`` pairs for the destroy-operator variants.

    Each name resolves to ``results/eval/<problem>/<name>/<task>/steps.csv``. Random Destroy
    leads so the baseline is first in the legend and color cycle. An empty string marks a
    variant with no evaluated run yet; a figure skips it.
    """
    return [
        ("Random", random),
        ("Joint + AR", joint_ar),
        ("Joint + NAR", joint_nar),
        ("Split + AR", split_ar),
        ("Split + NAR", split_nar),
    ]


# Title -> {problem, task, runs}. Fill a run with its eval config name once its steps.csv
# exists; keep the not-yet-evaluated variants commented with the name they will take.
EVAL_RUNS: dict[str, dict] = {
    "TSP20": {
        "problem": "tsp",
        "task": "tsp20",
        # NDS is only trained at sizes 50/100.
        "runs": variant_eval_runs(random="random_destroy"),
    },
    "TSP50": {
        "problem": "tsp",
        "task": "tsp50",
        "runs": variant_eval_runs(
            # joint_ar="nds_tsp_50+ar",             # config exists, not evaluated yet
            # joint_nar="nds_tsp_50+simple",        # config exists, not evaluated yet
            # split_nar="nds_tsp_50_split+simple",  # config exists, not evaluated yet
            random="random_destroy",
        ),
    },
    "TSP100": {
        "problem": "tsp",
        "task": "tsp100",
        "runs": variant_eval_runs(
            joint_ar="nds_tsp_100+ar",
            # joint_nar="nds_tsp_100+simple",     # config exists, not evaluated yet
            # split_ar="nds_tsp_100_split+ar",    # no config yet
            split_nar="nds_tsp_100_split+simple",
            random="random_destroy",
        ),
    },
    "CVRP20": {
        "problem": "cvrp",
        "task": "cvrp20",
        # NDS is only trained at sizes 50/100.
        "runs": variant_eval_runs(random="random_destroy"),
    },
    "CVRP50": {
        "problem": "cvrp",
        "task": "cvrp50",
        "runs": variant_eval_runs(
            # joint_ar="nds_cvrp_50+ar",             # config exists, not evaluated yet
            # joint_nar="nds_cvrp_50+simple",        # config exists, not evaluated yet
            # split_nar="nds_cvrp_50_split+simple",  # config exists, not evaluated yet
            random="random_destroy",
        ),
    },
    "CVRP100": {
        "problem": "cvrp",
        "task": "cvrp100",
        # Original-NDS baselines exist too (nds_original_cvrp_100[+nowarmup]); they are
        # a separate comparison and left out of the variant sweep here.
        "runs": variant_eval_runs(
            joint_ar="nds_cvrp_100+ar",
            joint_nar="nds_cvrp_100+simple",
            # split_ar="nds_cvrp_100_split+ar",  # config exists, not evaluated yet
            split_nar="nds_cvrp_100_split+simple",
            random="random_destroy",
        ),
    },
    "CVRP200": {
        "problem": "cvrp",
        "task": "cvrp200",
        # NDS is trained at size 100 but evaluated here for generalisation to size 200,
        # so the runs reuse the size-100 methods with the cvrp200 task.
        "runs": variant_eval_runs(
            # joint_ar="nds_cvrp_100+ar",             # not evaluated on cvrp200 yet
            # joint_nar="nds_cvrp_100+simple",        # not evaluated on cvrp200 yet
            # split_ar="nds_cvrp_100_split+ar",       # config exists, not evaluated
            # split_nar="nds_cvrp_100_split+simple",  # not evaluated on cvrp200 yet
            random="random_destroy",
        ),
    },
    "CMSTP20": {
        "problem": "cmstp",
        "task": "cmstp20",
        # NDS is only trained at sizes 50/100.
        "runs": variant_eval_runs(random="random_destroy"),
    },
    "CMSTP50": {
        "problem": "cmstp",
        "task": "cmstp50",
        "runs": variant_eval_runs(
            # joint_ar="nds_cmstp_50+ar",              # config exists, not evaluated yet
            # joint_nar="nds_cmstp_50+simple",         # config exists, not evaluated yet
            # split_ar="nds_cmstp_50_split+ar",        # config exists, not evaluated yet
            # split_nar="nds_cmstp_50_split+simple",   # config exists, not evaluated yet
            random="random_destroy",
        ),
    },
    "CMSTP100": {
        "problem": "cmstp",
        "task": "cmstp100",
        "runs": variant_eval_runs(
            # joint_ar="nds_cmstp_100+ar",             # config exists, not evaluated yet
            # joint_nar="nds_cmstp_100+simple",        # config exists, not evaluated yet
            # split_ar="nds_cmstp_100_split+ar",       # config exists, not evaluated yet
            # split_nar="nds_cmstp_100_split+simple",  # config exists, not evaluated yet
            random="random_destroy",
        ),
    },
}
