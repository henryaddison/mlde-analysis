eval_vars = ["pr", "tmean150cm", "relhum150cm", "swbgt"]
dataset_configs = {
    "CPM": "bham64_ccpm-4x_12em_mv",
    "GCM": "bham64_gcm-4x_12em_psl-sphum4th-temp4th-vort4th_pr",
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
                    "fq_model_id": "score-sde/subvpsde/ukcp_local_rs_12em_cncsnpp_continuous/bham-4x_12em_PslS4T4V4_Rs",
                    "checkpoint": "epoch_20",
                    "dataset": "bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_relhum150cm",
                    "input_xfm": "bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_relhum150cm-stan",
                    "variables": ["relhum150cm"],
                },
                {
                    "fq_model_id": "score-sde/subvpsde/ukcp_local_ts_12em_cncsnpp_continuous/bham-4x_12em_PslS4T4V4_Ts",
                    "checkpoint": "epoch_20",
                    "dataset": "bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_tmean150cm",
                    "input_xfm": "bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_tmean150cm-stan",
                    "variables": ["tmean150cm"],
                },
                {
                    "fq_model_id": "score-sde/subvpsde/xarray_12em_cncsnpp_continuous/bham-4x_12em_PslS4T4V4_random-season-IstanTsqrturrecen-no-loc-spec",
                    "checkpoint": "epoch_20",
                    "dataset": "bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr",
                    "input_xfm": "bham64_ccpm-4x_12em_psl-sphum4th-temp4th-vort4th_pr-stan",
                    "variables": ["pr"],
                },
            ],
            "label": "sa-cCPM",
            "deterministic": False,
            "PSD": True,
            "color": "blue",
            "order": 10,
            "CCS": True,
        },
        {
            "sample_specs": [
                {
                    "fq_model_id": "score-sde/subvpsde/ukcp_local_mv_12em_cncsnpp_continuous/bham-4x_12em_PslS4T4V4_PrRsTs",
                    "checkpoint": "epoch_20",
                    "dataset": "bham64_ccpm-4x_12em_mv",
                    "input_xfm": "bham64_ccpm-4x_12em_mv-stan",
                    "variables": ["pr", "tmean150cm", "relhum150cm"],
                },
            ],
            "label": "mv-cCPM",
            "deterministic": False,
            "PSD": True,
            "color": "cyan",
            "order": 11,
            "CCS": True,
        },
    ],
    # "GCM": []
}

derived_variables_config = {
    "swbgt": [
        "mlde_analysis.derived_variables.swbgt",
        {"temp": "tmean150cm", "rh": "relhum150cm"},
    ]
}

example_percentiles = {
    "CPM": {
        "JJA Hot": {"percentile": 0.8, "variable": "tmean150cm", "season": "JJA"},
        "DJF Humid": {"percentile": 0.8, "variable": "relhum150cm", "season": "DJF"},
        "DJF Cold": {"percentile": 0.2, "variable": "tmean150cm", "season": "DJF"},
    },
}
example_overrides = {
    "CPM": {
        "JJA Convective": ["01", "1993-08-01 12:00:00"],
    },
    "GCM": {},
}
example_inputs = ["vorticity850"]
n_samples_per_example = 1

desc = """
Describe in more detail the models being compared
"""
