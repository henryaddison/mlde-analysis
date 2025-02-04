import xskillscore as xs


def resample_examples(da, niterations):
    return xs.resampling.resample_iterations_idx(
        da.stack(member=["ensemble_member", "time"]),
        niterations,
        "member",
        replace=True,
    ).unstack("member")
