eval_vars = ["pr", "tasmax"]
target_sim_key = "RCM"
exclude_days = 0  # Number of days to exclude from each start of each season
split = "val"
ensemble_members = [
    "01",
]
samples_per_run = 1
alps_dataset_configs = {
    "RCM": "ALPS_domain-Emulator_hist_future-CNRMCM5-perfect",
}
alps_sample_configs = {
    "RCM": [
        {
            "sample_specs": [
                {
                    "fq_model_id": "mlde/score-sde/subvpsde/cordex_ml_mv_hist_fut_alps_cncsnpp_continuous/no_static_rcmgem",
                    "checkpoint": "epoch_200",
                    "dataset": "ALPS_domain-Emulator_hist_future-CNRMCM5-perfect",
                    "input_xfm": "ALPS_domain-Emulator_hist_future-CNRMCM5-perfect-stan",
                    "variables": ["pr", "tasmax"],
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
                    "variables": ["pr", "tasmax"],
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

sa_dataset_configs = {
    "RCM": "SA_domain-Emulator_hist_future-ACCESSCM2-perfect",
}
sa_sample_configs = {
    "RCM": [
        {
            "sample_specs": [
                {
                    "fq_model_id": "mlde/deterministic/cordex_ml_pr_hist_fut_sa_unet/no_static_rcmgem",
                    "checkpoint": "epoch_100",
                    "dataset": "SA_domain-Emulator_hist_future-ACCESSCM2-perfect",
                    "input_xfm": "SA_domain-Emulator_hist_future-ACCESSCM2-perfect-stan",
                    "variables": ["pr"],
                    "config_hash": "2f53e4b6e015c5f9",
                },
            ],
            "label": "SA U-Net",
            "deterministic": True,
            "PSD": True,
            "color": "tab:orange",
            "order": 20,
            "CCS": True,
        },
    ],
}

nz_dataset_configs = {
    "RCM": "NZ_domain-Emulator_hist_future-ACCESSCM2-perfect",
}
nz_sample_configs = {
    "RCM": [
        {
            "sample_specs": [
                {
                    "fq_model_id": "mlde/deterministic/cordex_ml_pr_hist_fut_nz_unet/no_static_rcmgem",
                    "checkpoint": "epoch_120",
                    "dataset": "NZ_domain-Emulator_hist_future-ACCESSCM2-perfect",
                    "input_xfm": "NZ_domain-Emulator_hist_future-ACCESSCM2-perfect-stan",
                    "variables": ["pr"],
                    "config_hash": "8659e730fb341654",
                },
            ],
            "label": "NZ U-Net",
            "deterministic": True,
            "PSD": True,
            "color": "tab:orange",
            "order": 20,
            "CCS": True,
        },
    ],
}

dataset_configs = alps_dataset_configs
sample_configs = alps_sample_configs

derived_variables_config = {}

example_percentiles = {
    "RCM": {
        "DJF Wet": {"percentile": 0.8, "variable": "pr", "season": "DJF"},
        "DJF Wettest": {"percentile": 1.0, "variable": "pr", "season": "DJF"},
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
