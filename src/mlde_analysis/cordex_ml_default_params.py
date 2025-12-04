eval_vars = ["pr"]
target_sim_key = "RCM"
dataset_configs = {
    "RCM": "ALPS_domain-Emulator_hist_future-CNRMCM5-perfect",
}
exclude_days = 0  # Number of days to exclude from each start of each season
split = "val"
ensemble_members = [
    "01",
]
samples_per_run = 1
sample_configs = {
    "RCM": [
        {
            "sample_specs": [
                {
                    "fq_model_id": "mlde/score-sde/subvpsde/cordex_ml_mv_hist_fut_alps_cncsnpp_continuous/no_static_rcmgem",
                    "checkpoint": "epoch_200",
                    "dataset": "ALPS_domain-Emulator_hist_future-CNRMCM5-perfect",
                    "input_xfm": "ALPS_domain-Emulator_hist_future-CNRMCM5-perfect-stan",
                    "variables": ["pr"],
                    "config_hash": "cfb4519edcc06af1",
                },
            ],
            "label": "ALPS-RCMGEM",
            "deterministic": False,
            "PSD": True,
            "color": "tab:blue",
            "order": 10,
            "CCS": True,
        },
        {
            "sample_specs": [
                {
                    "fq_model_id": "mlde/deterministic/cordex_ml_mv_hist_fut_alps_unet/no_static_rcmgem",
                    "checkpoint": "epoch_200",
                    "dataset": "ALPS_domain-Emulator_hist_future-CNRMCM5-perfect",
                    "input_xfm": "ALPS_domain-Emulator_hist_future-CNRMCM5-perfect-stan",
                    "variables": ["pr"],
                    "config_hash": "b4b44a4d9b9c2b1a",
                },
            ],
            "label": "ALPS-U-Net",
            "deterministic": True,
            "PSD": True,
            "color": "tab:orange",
            "order": 20,
            "CCS": True,
        },
    ],
}

derived_variables_config = {}

example_percentiles = {
    "RCM": {
        "DJF Wet": {"percentile": 0.8, "variable": "pr", "season": "DJF"},
        "DJF Wettest": {"percentile": 0.2, "variable": "pr", "season": "DJF"},
        "JJA Wet": {"percentile": 0.8, "variable": "pr", "season": "JJA"},
        "JJA Wettest": {"percentile": 1.0, "variable": "pr", "season": "JJA"},
    },
}
example_overrides = {
    "RCM": {},
}
example_inputs = []
examples_sample_idxs = 1


desc = """
Describe in more detail the models being compared
"""

# bootstrapping
niterations = 5
bootstrap_configs = {
    "niterations": {
        "spread-error": 10,
    }
}

# eval@60km

sample_configs_at_60km = []
dataset_configs_at_60km = {}
