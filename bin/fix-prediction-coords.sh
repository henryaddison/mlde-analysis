#!/bin/bash

set -euo pipefail

# 60y 8.8km
for p in /gws/ssde/j25a/furflex/henrya/projects/furflex/workdirs/cpmgem-subdaily/ew_pr_cpmgem-daily_preset/latte-{b,xl}8/4rtrecen-2000steps/samples/*/*/val/*/*/r001i1p0*/predictions.zarr; do
  pixi run python bin/fix-prediction-coords.py ${p} engwales_ccpm-4x-cpmgem_12em_1hr_pr_preset_v2
done

# 100y 8.8km
for p in /gws/ssde/j25a/furflex/henrya/projects/furflex/workdirs/cpmgem-subdaily/v3_engwales_100y_1hr_pr/*/*/samples/*/*/val/*/*/r001i1p0*/predictions.zarr; do
  pixi run python bin/fix-prediction-coords.py ${p} v3_engwales_ccpm-4x_100x12em_1hr_pr
done

# 5km
for p in /gws/ssde/j25a/furflex/henrya/projects/furflex/workdirs/cpmgem-subdaily/v3_engwales_100y_5km_1hr_pr/*/*/samples/*/*/val/*/*/r001i1p0*/predictions.zarr; do
  pixi run python bin/fix-prediction-coords.py ${p} v3_engwales_ccpm-5km_100x12em_1hr_pr
done
