eval_vars = ["pr"]
dataset_configs = {
    "CPM": "demo-ccpm_pr",
    "GCM": "demo-gcm_pr",
}
split = "val"
ensemble_members = [
    "01",
    # "04",
    # "05",
    # "06",
    # "07",
    # "08",
    # "09",
    # "10",
    # "11",
    # "12",
    # "13",
    # "15",
]
samples_per_run = 3
sample_configs = {
    "CPM": [
        {
            "sample_specs": [
                {
                    "fq_model_id": "score-sde/demo-cpmgem-pr",
                    "checkpoint": "epoch_20",
                    "dataset": "demo-ccpm_pr",
                    "input_xfm": "demo-ccpm_pr-stan",
                    "variables": ["pr"],
                },
            ],
            "label": "sa-cCPM",
            "deterministic": False,
            "PSD": True,
            "color": "tab:blue",
            "order": 10,
            "CCS": True,
        },
        {
            "sample_specs": [
                {
                    "fq_model_id": "score-sde/demo-cpmgem-mv",
                    "checkpoint": "epoch_20",
                    "dataset": "demo-ccpm_mv",
                    "input_xfm": "demo-ccpm_mv-stan",
                    "variables": ["pr"],
                },
            ],
            "label": "mv-cCPM",
            "deterministic": False,
            "PSD": True,
            "color": "tab:orange",
            "order": 11,
            "CCS": True,
        },
        {
            "sample_specs": [
                {
                    "fq_model_id": "u-net/bham-4x_12em_pSTV",
                    "checkpoint": "epoch_100",
                    "dataset": "demo-ccpm_pr",
                    "input_xfm": "demo-ccpm_pr-stan",
                    "variables": ["pr"],
                },
            ],
            "label": "U-Net-cCPM",
            "deterministic": True,
            "PSD": True,
            "color": "tab:red",
            "order": 5,
            "CCS": False,
        },
        {
            "label": "cCPM Bilinear",
            "sample_specs": [
                {
                    "fq_model_id": "id-linpr",
                    "checkpoint": "epoch_0",
                    "dataset": "demo-ccpm_pr",
                    "input_xfm": "none",
                    "variables": ["pr"],
                },
            ],
            "deterministic": True,
            "color": "tab:grey",
            "CCS": False,
            "UQ": False,
            "order": 1,
        },
    ],
    "GCM": [
        {
            "sample_specs": [
                {
                    "fq_model_id": "score-sde/demo-cpmgem-pr",
                    "checkpoint": "epoch_20",
                    "dataset": "demo-gcm_pr",
                    "input_xfm": "demo-gcm_pr-pixelmmsstan",
                    "variables": ["pr"],
                },
            ],
            "label": "sa-GCM",
            "deterministic": False,
            "PSD": True,
            "color": "tab:cyan",
            "order": 100,
            "CCS": True,
        },
    ],
}

derived_variables_config = {}

example_percentiles = {
    "CPM": {
        "DJF Wet": {"percentile": 0.8, "variable": "pr", "season": "DJF"},
        "DJF Wettest": {"percentile": 0.2, "variable": "pr", "season": "DJF"},
        "JJA Wet": {"percentile": 0.8, "variable": "pr", "season": "JJA"},
        "JJA Wettest": {"percentile": 1.0, "variable": "pr", "season": "JJA"},
    },
}
example_overrides = {
    "CPM": {
        "JJA Wet": ["01", "1993-08-01 12:00:00"],
    },
    "GCM": {},
}
example_inputs = ["vorticity850"]
n_samples_per_example = 2

desc = """
Describe in more detail the models being compared
"""

# bootstrapping
niterations = 5
