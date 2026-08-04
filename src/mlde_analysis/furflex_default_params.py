desc = """
Describe in more detail the models being compared
"""
eval_vars = ["pr"]
target_sim_key = "CPM"
dataset_configs = {
    "CPM": "engwales_ccpm-4x-cpmgem_12em_1hr_pr_preset_v2",
}
exclude_days = 0  # Number of days to exclude from each start of each season
split = "val"
ensemble_members = [
    "r001i1p00000",
    "r001i1p01113",
    "r001i1p01554",
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
sample_configs = {
    "CPM": [
        {
            "sample_specs": [
                {
                    "fq_model_id": "ew_pr_cpmgem-daily_preset/latte-b8/4rtrecen",
                    "checkpoint": "0150",
                    "dataset": "engwales_ccpm-4x-cpmgem_12em_1hr_pr_preset_v2",
                    "input_xfm": "",
                    "variables": ["pr"],
                    "config_hash": "1000-steps",
                },
            ],
            "label": "e0150",
            "deterministic": False,
            "PSD": True,
            "color": "tab:blue",
            "CCS": True,
        },
        {
            "sample_specs": [
                {
                    "fq_model_id": "ew_pr_cpmgem-daily_preset/latte-b8/4rtrecen",
                    "checkpoint": "0300",
                    "dataset": "engwales_ccpm-4x-cpmgem_12em_1hr_pr_preset_v2",
                    "input_xfm": "",
                    "variables": ["pr"],
                    "config_hash": "1000-steps",
                },
            ],
            "label": "e0300",
            "deterministic": False,
            "PSD": True,
            "color": "tab:orange",
            "CCS": True,
        },
        {
            "sample_specs": [
                {
                    "fq_model_id": "ew_pr_cpmgem-daily_preset/latte-b8/4rtrecen",
                    "checkpoint": "0500",
                    "dataset": "engwales_ccpm-4x-cpmgem_12em_1hr_pr_preset_v2",
                    "input_xfm": "",
                    "variables": ["pr"],
                    "config_hash": "1000-steps",
                },
            ],
            "label": "e0500",
            "deterministic": False,
            "PSD": True,
            "color": "tab:red",
            "CCS": True,
        },
    ],
}

derived_variables_config = {}

examples_to_plot = {
    "CPM": {
        "band": {
            "times": ["2080-03-01", "2080-03-02"],
            "query": {"ensemble_member": "r001i1p00000"},
        },
        "conv": {
            "times": ["2029-06-23", "2029-06-24"],
            "query": {"ensemble_member": "r001i1p02868"},
        },
    },
}
