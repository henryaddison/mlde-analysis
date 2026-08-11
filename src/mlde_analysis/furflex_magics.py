import cf_xarray as cfxr  # noqa: F401
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.shellapp import InteractiveShellApp
from IPython.core.magic_arguments import (
    argument,
    magic_arguments,
    parse_argstring,
)  # type: ignore
from mlde_analysis.furflex_data import prep_eval_data
import xarray as xr


@magics_class
class LoadEvalData(Magics):

    @magic_arguments()
    @argument(
        "-c",
        "--coarsen",
        action="store",
        default=None,
        type=int,
        help="An optional argument to coarsen time dim.",
    )
    @line_magic
    def load_eval_data(self, line):
        args = parse_argstring(self.load_eval_data, line)
        eval_vars = self.shell.user_ns["eval_vars"]
        target_sim_key = self.shell.user_ns["target_sim_key"]
        eval_ds, models = prep_eval_data(
            self.shell.user_ns["sample_configs"],
            self.shell.user_ns["dataset_configs"],
            self.shell.user_ns["derived_variables_config"],
            eval_vars,
            self.shell.user_ns["split"],
            exclude_days=self.shell.user_ns["exclude_days"],
            ensemble_members=self.shell.user_ns["ensemble_members"],
            samples_per_run=self.shell.user_ns["samples_per_run"],
            coarsen_time=args.coarsen,
            target_sim_key=target_sim_key,
        )

        target_sim_das = {
            var: eval_ds[target_sim_key].cf[f"target_{var}"] for var in eval_vars
        }

        pred_das = {
            var: xr.concat(
                [ds[f"pred_{var}"] for ds in eval_ds.values()],
                dim="model",
            )
            for var in eval_vars
        }

        var_das = {
            var: xr.merge([pred_das[var], target_sim_das[var]], compat="no_conflicts")
            for var in eval_vars
        }

        modellabel2spec = {
            model_label: {"source": source} | model_spec
            for source, source_models in models.items()
            for model_label, model_spec in source_models.items()
        } | {target_sim_key: {"source": target_sim_key, "color": "black"}}

        return eval_ds, models, target_sim_das, pred_das, var_das, modellabel2spec


def load_ipython_extension(ipython: InteractiveShellApp):
    ipython.register_magics(LoadEvalData)
