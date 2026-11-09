from collections import defaultdict
from itertools import groupby, zip_longest
from operator import attrgetter
from pathlib import Path
import typer


class Sample:
    def __init__(self, p):
        self.path = p
        self.em = p.parts[0]
        self.config_desc = p.parts[1]
        self.pred_id = p.parts[2]

    def __repr__(self):
        return f"Sample({self.config_desc}, {self.em}, {self.pred_id})"

    def new_path(self, ss_id):
        return Path(self.config_desc, ss_id, self.em)


app = typer.Typer()


@app.command()
def main(emu_samples_dir: Path):
    new_emu_samples_dir = (
        emu_samples_dir.parent / "new-samples"
    )  # store moved samples in a new folder so can pick up where left off if fails
    emu_sample_set_roots = list(emu_samples_dir.glob("*/*/val"))

    sample_sets = {}

    for emu_sample_set_root in emu_sample_set_roots:
        # for each checkpoint and dataset split, gathers samples and sort according to config_desc
        sample_sets[emu_sample_set_root] = []
        samples = sorted(
            map(
                lambda p: Sample(p.relative_to(emu_sample_set_root).parent),
                emu_sample_set_root.glob("*/*/*/predictions.zarr"),
            ),
            key=attrgetter("config_desc", "em"),
        )

        # group sorted samples according to config_desc and create list of pred_ids for each ensemble_member
        groups = []
        for k, g in groupby(samples, key=lambda s: s.config_desc):
            output = defaultdict(list)
            for s in g:
                output[s.em].append(s)

            groups.append(output)

        # create samples sets by zipping together Samples from each ensemble member
        for group in groups:
            sample_sets[emu_sample_set_root].extend(list(zip_longest(*group.values())))

    print("#!/bin/bash")
    print("set -euo pipefail")
    print("")

    for emu_sample_set_root in emu_sample_set_roots:
        for ss in sample_sets[emu_sample_set_root]:
            # remove Nones from a sample set (where not all ensemble members have samples)
            ss = list(filter(lambda e: e is not None, ss))
            # pick a pred_id to be the sample set id

            ss_id = ss[0].pred_id

            # move samples in a set to new path
            for s in ss:
                current_s_path = emu_sample_set_root / s.path
                new_s_path = (
                    new_emu_samples_dir
                    / emu_sample_set_root.relative_to(emu_samples_dir)
                    / s.new_path(ss_id)
                )
                print(f"echo Moving {current_s_path} to {new_s_path}")
                print(f"echo")
                print(f"mkdir -p {new_s_path.parent}")
                print(f"mv {current_s_path} {new_s_path}")
                print("")


if __name__ == "__main__":
    app()
