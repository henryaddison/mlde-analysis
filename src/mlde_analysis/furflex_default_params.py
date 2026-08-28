# Available contexts: "5km-100y", "8.8km-60y", or "8.8km-100y"
# use to select the appropriate default configs
# CONTEXT = "5km-100y"
CONTEXT = "8.8km-60y"
# CONTEXT = "8.8km-100y"

desc = """
Describe in more detail the models being compared
"""
eval_vars = ["pr"]
target_sim_key = "CPM"
if CONTEXT == "5km-100y":
    dataset_configs = {
        "CPM": "v3_engwales_ccpm-5km_100x12em_1hr_pr",
    }
    sample_configs = {
        "CPM": [
            {
                "sample_specs": [
                    {
                        "fq_model_id": "v3_engwales_100y_5km_1hr_pr/latte-b8/4rtrecen-2000steps",
                        "checkpoint": "0025",
                        "dataset": "v3_engwales_ccpm-5km_100x12em_1hr_pr",
                        "input_xfm": "",
                        "variables": ["pr"],
                        "config_hash": "1000-steps",
                    },
                ],
                "label": "e0025",
                "deterministic": False,
                "PSD": True,
                "color": "tab:blue",
                "CCS": True,
            },
        ],
    }
    examples_to_plot = {
        "CPM": {
            "v3": {
                "times": ["1989-12-01", "1989-12-01"],
                "query": {"ensemble_member": "r001i1p00000"},
            },
        },
    }
elif CONTEXT == "8.8km-60y":
    dataset_configs = {
        "CPM": "engwales_ccpm-4x-cpmgem_12em_1hr_pr_preset_v2",
        "GCM": "engwales_gcm-4x-cpmgem_12em_1hr_pr_preset_v2",
    }
    sample_configs = {
        "CPM": [
            {
                "sample_specs": [
                    {
                        "fq_model_id": "ew_pr_cpmgem-daily_preset/latte-b8/4rtrecen-2000steps",
                        "checkpoint": "0500",
                        "dataset": "engwales_ccpm-4x-cpmgem_12em_1hr_pr_preset_v2",
                        "input_xfm": "",
                        "variables": ["pr"],
                        "config_hash": "2000-steps",
                    },
                ],
                "label": "e0500 cCPM",
                "deterministic": False,
                "PSD": True,
                "color": "tab:blue",
                "CCS": True,
            },
        ],
        "GCM": [
            {
                "sample_specs": [
                    {
                        "fq_model_id": "ew_pr_cpmgem-daily_preset/latte-b8/4rtrecen-2000steps",
                        "checkpoint": "0500",
                        "dataset": "engwales_gcm-4x-cpmgem_12em_1hr_pr_preset_v2",
                        "input_xfm": "",
                        "variables": ["pr"],
                        "config_hash": "2000-steps",
                    },
                ],
                "label": "e0500 GCM",
                "deterministic": False,
                "PSD": True,
                "color": "tab:orange",
                "CCS": True,
            },
        ],
    }
    examples_to_plot = {
        "CPM": {
            "band": {
                "times": ["2080-03-01", "2080-03-02"],
                "query": {"ensemble_member": "r001i1p00000"},
            },
            # "conv": {
            #     "times": ["2029-06-23", "2029-06-24"],
            #     "query": {"ensemble_member": "r001i1p02868"},
            # },
        },
        "GCM": {
            "band": {
                "times": ["2080-03-01", "2080-03-02"],
                "query": {"ensemble_member": "r001i1p00000"},
            },
            # "conv": {
            #     "times": ["2029-06-23", "2029-06-24"],
            #     "query": {"ensemble_member": "r001i1p02868"},
            # },
        },
    }
elif CONTEXT == "8.8km-100y":
    dataset_configs = {
        "CPM": "v3_engwales_ccpm-4x_100x12em_1hr_pr",
    }
    sample_configs = {}
    examples_to_plot = {
        "CPM": {
            "v3": {
                "times": ["1989-12-01", "1989-12-01"],
                "query": {"ensemble_member": "r001i1p00000"},
            },
        },
    }
else:
    raise ValueError(f"Unknown context: {CONTEXT}")
exclude_days = 0  # Number of days to exclude from each start of each season
split = "val"
ensemble_members = [
    "r001i1p00000",
    # "r001i1p01113",
    # "r001i1p01554",
    # "r001i1p01649",
    # "r001i1p01843",
    # "r001i1p01935",
    # "r001i1p02868",
    # "r001i1p02123",
    # "r001i1p02242",
    # "r001i1p02305",
    # "r001i1p02335",
    # "r001i1p02491",
]
samples_per_run = 1

derived_variables_config = {}
