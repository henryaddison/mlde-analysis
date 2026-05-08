desc = """
Describe in more detail the models being compared
"""
eval_vars = ["pr"]
target_sim_key = "CPM"
dataset_configs = {
    "CPM": "engwales_ccpm-4x-cpmgem_12em_1hr_pr_preset",
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
                    "fq_model_id": "latte-b8-cpmgem-daily-preset",
                    "checkpoint": "0150",
                    "dataset": "engwales_ccpm-4x-cpmgem_12em_1hr_pr_preset",
                    "input_xfm": "",
                    "variables": ["pr"],
                },
            ],
            "label": "e0150",
            "deterministic": False,
            "PSD": True,
            "color": "tab:blue",
            "order": 10,
            "CCS": True,
        },
    ],
}

derived_variables_config = {}

examples_to_plot = {
    "CPM": {
        "band": {
            "times": ["2080-03-01", "2080-03-03"],
            "query": {"ensemble_member": "r001i1p00000"},
        },
    },
}
