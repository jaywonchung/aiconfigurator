# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

from aiconfigurator import __version__
from aiconfigurator.generator.api import generate_backend_config
from aiconfigurator.generator.cli_args import add_config_generation_cli, build_dynamo_config
from aiconfigurator.sdk import common, pareto_analysis, task
from aiconfigurator.sdk.pareto_analysis import (
    draw_pareto_to_string,
    get_best_configs_under_tpot_constraint,
)
from aiconfigurator.sdk.task import TaskConfig, TaskRunner
from aiconfigurator.sdk.utils import safe_mkdir
from aiconfigurator.cli.report_and_save import log_final_summary, save_results


logger = logging.getLogger(__name__)


def _build_common_cli_parser() -> argparse.ArgumentParser:
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the results.")
    common_parser.add_argument("--dump-all-configs", action="store_true", help="Dump all explored configurations (not just Pareto-optimal) to CSV files.")
    common_parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    add_config_generation_cli(common_parser, default_backend=common.BackendName.trtllm.value)
    return common_parser


def _add_default_mode_arguments(parser):
    parser.add_argument("--total_gpus", type=int, required=True, help="Total GPUs for deployment.")
    parser.add_argument("--model", choices=common.SupportedModels.keys(), type=str, required=True, help="Model name.")
    parser.add_argument("--system", type=str, required=True, help="Default system name.")
    parser.add_argument("--decode_system", type=str, default=None, help="System name for disagg decode workers. Defaults to --system if omitted.")
    parser.add_argument("--backend", choices=[backend.value for backend in common.BackendName], type=str, default=common.BackendName.trtllm.value, help="Backend name.")
    parser.add_argument("--backend_version", type=str, default=None, help="Backend database version. Default is latest")
    parser.add_argument("--isl", type=int, default=4000, help="Input sequence length.")
    parser.add_argument("--osl", type=int, default=1000, help="Output sequence length.")
    parser.add_argument("--ttft", type=float, default=1000.0, help="Time to first token in ms.")
    parser.add_argument("--tpot", type=float, default=20.0, help="Time per output token in ms.")

    # Power-related arguments
    parser.add_argument("--power-limits", type=str, default=None, help='Power limits to explore (W). Comma-separated list (e.g., "400,500,600,700") or single value. Defaults to max available power limit if not specified.')
    parser.add_argument("--prefill-power-limits", type=str, default=None, help='Power limits for prefill workers in disaggregated mode (W). Comma-separated list.')
    parser.add_argument("--decode-power-limits", type=str, default=None, help='Power limits for decode workers in disaggregated mode (W). Comma-separated list.')
    parser.add_argument("--power-budget", type=float, default=None, help='Total cluster power budget in Watts. Used to filter/highlight configurations in visualization.')


def _add_experiments_mode_arguments(parser):
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to a YAML file containing experiment definitions.")


def configure_parser(parser):
    common_cli_parser = _build_common_cli_parser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    default_parser = subparsers.add_parser("default", parents=[common_cli_parser], help="Run the default agg vs disagg comparison.")
    _add_default_mode_arguments(default_parser)

    help_text = "Run one or more experiments defined in a YAML file. Example: example.yaml"
    # an example yaml for demonstration
    example_yaml_path = os.path.join(os.path.dirname(__file__), "example.yaml")
    with open(example_yaml_path, "r") as f:
        example_yaml = yaml.safe_load(f)
    description = help_text + "\n\nExample:\n" + json.dumps(example_yaml, indent=2)

    experiments_parser = subparsers.add_parser(
        "exp", 
        parents=[common_cli_parser], 
        help=help_text,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_experiments_mode_arguments(experiments_parser)


def _build_default_task_configs(args) -> Dict[str, TaskConfig]:
    decode_system = args.decode_system or args.system

    # Parse power limits from comma-separated strings to lists
    power_limits = None
    if hasattr(args, 'power_limits') and args.power_limits:
        power_limits = [int(x.strip()) for x in args.power_limits.split(',')]

    prefill_power_limits = None
    if hasattr(args, 'prefill_power_limits') and args.prefill_power_limits:
        prefill_power_limits = [int(x.strip()) for x in args.prefill_power_limits.split(',')]

    decode_power_limits = None
    if hasattr(args, 'decode_power_limits') and args.decode_power_limits:
        decode_power_limits = [int(x.strip()) for x in args.decode_power_limits.split(',')]

    cluster_power_budget = None
    if hasattr(args, 'power_budget') and args.power_budget:
        cluster_power_budget = args.power_budget

    common_kwargs: Dict[str, Any] = {
        "model_name": args.model,
        "system_name": args.system,
        "backend_name": args.backend,
        "backend_version": args.backend_version,
        "total_gpus": args.total_gpus,
        "isl": args.isl,
        "osl": args.osl,
        "ttft": args.ttft,
        "tpot": args.tpot,
        "cluster_power_budget": cluster_power_budget,
    }

    # Agg mode uses power_limits
    agg_kwargs = dict(common_kwargs)
    agg_kwargs["power_limits"] = power_limits
    agg_task = TaskConfig(serving_mode="agg", **agg_kwargs)

    # Disagg mode uses prefill_power_limits and decode_power_limits
    disagg_kwargs = dict(common_kwargs)
    disagg_kwargs["decode_system_name"] = decode_system
    disagg_kwargs["prefill_power_limits"] = prefill_power_limits
    disagg_kwargs["decode_power_limits"] = decode_power_limits
    disagg_task = TaskConfig(serving_mode="disagg", **disagg_kwargs)

    return {"disagg": disagg_task, "agg": agg_task}


_EXPERIMENT_RESERVED_KEYS = {
    "mode",
    "serving_mode",
    "model_name",
    "system_name",
    "decode_system_name",
    "backend_name",
    "backend_version",
    "profiles",
    "isl",
    "osl",
    "ttft",
    "tpot",
    "enable_wide_ep",
    "total_gpus",
    "use_specific_quant_mode",
    "power_limits",
    "prefill_power_limits",
    "decode_power_limits",
    "cluster_power_budget",
}


def _build_yaml_config(exp_config: dict, config_section: dict) -> Optional[dict]:
    if not config_section:
        config_section = {
            key: copy.deepcopy(value)
            for key, value in exp_config.items()
            if key not in _EXPERIMENT_RESERVED_KEYS
        }
    if not config_section:
        return None

    yaml_config = {
        "mode": exp_config.get("mode", "patch"),
        "config": config_section,
    }

    return yaml_config


def _build_experiment_task_configs(args) -> Dict[str, TaskConfig]:
    try:
        with open(args.yaml_path, "r", encoding="utf-8") as fh:
            experiment_data = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.error("Error loading experiment YAML file '%s': %s", args.yaml_path, exc)
        raise SystemExit(1) from exc

    if not isinstance(experiment_data, dict):
        logger.error("Experiment YAML root must be a mapping.")
        raise SystemExit(1)

    order = experiment_data.get("exps")
    if isinstance(order, list):
        experiment_names = [name for name in order if name in experiment_data]
    else:
        experiment_names = [name for name in experiment_data.keys() if name != "exps"]

    task_configs: Dict[str, TaskConfig] = {}

    for exp_name in experiment_names:
        exp_config = experiment_data[exp_name]
        if not isinstance(exp_config, dict):
            logger.warning("Skipping experiment '%s': configuration is not a mapping.", exp_name)
            continue

        config_section = exp_config.get("config")
        if not isinstance(config_section, dict):
            config_section = {}
        else:
            config_section = copy.deepcopy(config_section)

        serving_mode = exp_config["serving_mode"]
        model_name = exp_config["model_name"]
        if serving_mode not in {"agg", "disagg"} or not model_name:
            logger.warning("Skipping experiment '%s': missing serving_mode or model_name.", exp_name)
            continue

        # system
        if serving_mode == "agg":
            inferred_system = exp_config["system_name"]
            inferred_decode_system = None
        else:
            inferred_system = exp_config["system_name"]
            inferred_decode_system = exp_config.get("decode_system_name") or inferred_system
        system_name = inferred_system
        if not system_name:
            logger.warning(
                "Skipping experiment '%s': no system name provided (provide system_name at the top level or inside worker config).",
                exp_name,
            )
            continue

        # backend, default to trtllm
        if serving_mode == "agg":
            backend_name = exp_config.get("backend_name") or common.BackendName.trtllm.value
            backend_version = exp_config.get("backend_version")
        else:
            backend_name = exp_config.get("backend_name") or common.BackendName.trtllm.value
            backend_version = exp_config.get("backend_version")

        total_gpus = exp_config.get("total_gpus")
        if total_gpus is None:
            logger.warning("Skipping experiment '%s': total_gpus not provided in YAML.", exp_name)
            continue

        task_kwargs: Dict[str, Any] = {
            "serving_mode": serving_mode,
            "model_name": model_name,
            "system_name": system_name,
            "backend_name": backend_name,
            "total_gpus": total_gpus,
            "profiles": exp_config.get("profiles", []),
        }

        if backend_version is not None:
            task_kwargs["backend_version"] = backend_version

        if serving_mode == "disagg":
            task_kwargs["decode_system_name"] = inferred_decode_system or system_name

        # Power-aware overrides may be provided at the experiment level.  These
        # should not flow into the YAML patch because the execution logic reads
        # them directly from the TaskConfig instance.
        def _normalize_limits(value: Any, field_name: str) -> Optional[list[int]]:
            if value is None:
                return None
            if isinstance(value, list):
                normalized: list[int] = []
                for item in value:
                    if isinstance(item, (int, float)) and not isinstance(item, bool):
                        normalized.append(int(item))
                    elif isinstance(item, str):
                        item = item.strip()
                        if not item:
                            continue
                        try:
                            normalized.append(int(float(item)))
                        except ValueError as exc:
                            logger.warning(
                                "Skipping invalid entry '%s' for %s in experiment '%s'",
                                item,
                                field_name,
                                exp_name,
                            )
                            continue
                    else:
                        logger.warning(
                            "Skipping unsupported value %s (type %s) for %s in experiment '%s'",
                            item,
                            type(item).__name__,
                            field_name,
                            exp_name,
                        )
                return normalized or None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return [int(value)]
            if isinstance(value, str):
                parts = [part.strip() for part in value.split(",")]
                parsed = []
                for part in parts:
                    if not part:
                        continue
                    try:
                        parsed.append(int(float(part)))
                    except ValueError:
                        logger.warning(
                            "Skipping invalid entry '%s' for %s in experiment '%s'",
                            part,
                            field_name,
                            exp_name,
                        )
                return parsed or None
            logger.warning(
                "Ignoring %s value of unsupported type %s in experiment '%s'",
                field_name,
                type(value).__name__,
                exp_name,
            )
            return None

        power_limits = _normalize_limits(exp_config.get("power_limits"), "power_limits")
        if power_limits is not None:
            task_kwargs["power_limits"] = power_limits

        prefill_power_limits = _normalize_limits(
            exp_config.get("prefill_power_limits"), "prefill_power_limits"
        )
        if prefill_power_limits is not None:
            task_kwargs["prefill_power_limits"] = prefill_power_limits

        decode_power_limits = _normalize_limits(
            exp_config.get("decode_power_limits"), "decode_power_limits"
        )
        if decode_power_limits is not None:
            task_kwargs["decode_power_limits"] = decode_power_limits

        cluster_power_budget = exp_config.get("cluster_power_budget")
        if cluster_power_budget is not None:
            try:
                task_kwargs["cluster_power_budget"] = float(cluster_power_budget)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring cluster_power_budget value '%s' for experiment '%s'",
                    cluster_power_budget,
                    exp_name,
                )

        # Per-experiment overrides for runtime numeric parameters if provided at top level
        for numeric_key in ("isl", "osl", "ttft", "tpot"):
            if numeric_key in exp_config:
                task_kwargs[numeric_key] = exp_config[numeric_key]

        if "enable_wide_ep" in exp_config:
            task_kwargs["enable_wide_ep"] = exp_config["enable_wide_ep"]
        if "use_specific_quant_mode" in exp_config:
            task_kwargs["use_specific_quant_mode"] = exp_config["use_specific_quant_mode"]

        yaml_config = _build_yaml_config(exp_config, config_section)
        if yaml_config:
            task_kwargs["yaml_config"] = yaml_config

        try:
            task_configs[exp_name] = TaskConfig(**task_kwargs)
        except Exception as exc:
            logger.error("Failed to build TaskConfig for experiment '%s': %s", exp_name, exc)
            logger.exception("Full traceback")

    if not task_configs:
        logger.error("No valid experiments found in '%s'.", args.yaml_path)
        raise SystemExit(1)

    return task_configs


def _execute_task_configs(
    task_configs: Dict[str, TaskConfig],
    mode: str,
) -> Tuple[str, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, float]]:
    """Execute the task configs and return the chosen experiment, best configs, pareto fronts, all explored configs, and best throughputs."""
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    start_time = time.time()
    runner = TaskRunner()

    # TODO, can run in parallel
    for exp_name, task_config in task_configs.items():
        try:
            logger.info("Starting experiment: %s", exp_name)
            logger.info("Task config: %s", task_config.pretty())
            task_result = runner.run(task_config)
            pareto_df = task_result["pareto_df"]
            pareto_frontier_df = task_result["pareto_frontier_df"]
            if pareto_frontier_df is not None and not pareto_frontier_df.empty:
                results[exp_name] = task_result
                logger.info("Experiment %s completed with %d results.", exp_name, len(pareto_frontier_df))
            else:
                logger.warning("Experiment %s returned no results.", exp_name)
        except Exception as exc:
            logger.error("Error running experiment %s: %s", exp_name, exc)
            logger.exception("Full traceback")

    if len(results) < 1:
        logger.error("No successful experiment runs to compare.")
        raise SystemExit(1)

    best_configs: Dict[str, pd.DataFrame] = {}
    best_throughputs: Dict[str, float] = {}
    pareto_fronts: Dict[str, Optional[pd.DataFrame]] = {}
    all_explored_configs: Dict[str, Optional[pd.DataFrame]] = {}
    for name, task_result in results.items():
        pareto_df = task_result["pareto_df"]
        pareto_frontier_df = task_result["pareto_frontier_df"]
        target_tpot = task_configs[name].config.runtime_config.tpot
        total_gpus = getattr(task_configs[name], "total_gpus", None) or 0
        group_by_key = "(d)parallel" if task_configs[name].serving_mode == "disagg" else "parallel"
        best_config_df = get_best_configs_under_tpot_constraint( # based on all data points
            total_gpus=total_gpus,
            pareto_df=pareto_df,
            target_tpot=target_tpot,
            top_n=5,
            group_by=group_by_key,
        )
        best_configs[name] = best_config_df
        pareto_fronts[name] = pareto_frontier_df
        all_explored_configs[name] = pareto_df
        if not best_config_df.empty:
            best_throughputs[name] = best_config_df['tokens/s/gpu_cluster'].values[0]
        else:
            best_throughputs[name] = 0.0

    chosen_exp = max(best_throughputs, key=best_throughputs.get) if best_throughputs else "none"

    log_final_summary(
        chosen_exp=chosen_exp, # for summary
        best_throughputs=best_throughputs, # for summary
        best_configs=best_configs, # for table
        pareto_fronts=pareto_fronts, # for plotting
        task_configs=task_configs, # for info in summary
        mode=mode,
    )

    end_time = time.time()
    logger.info("All experiments completed in %.2f seconds", end_time - start_time)

    return chosen_exp, best_configs, pareto_fronts, all_explored_configs, best_throughputs


def main(args):
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

    logger.info(f"Loading Dynamo AIConfigurator version: {__version__}")

    if args.mode == "default":
        task_configs = _build_default_task_configs(args)
    elif args.mode == "exp":
        task_configs = _build_experiment_task_configs(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")

    chosen_exp, best_configs, pareto_fronts, all_explored_configs, best_throughputs = _execute_task_configs(
        task_configs,
        args.mode,
    )

    if args.save_dir:
        save_results(
            args=args,
            best_configs=best_configs,
            pareto_fronts=pareto_fronts,
            all_explored_configs=all_explored_configs if args.dump_all_configs else None,
            task_configs=task_configs,
            save_dir=args.save_dir,
            generated_backend_version=args.generated_config_version,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamo AIConfigurator for Disaggregated Serving Deployment")
    configure_parser(parser)
    args = parser.parse_args()
    main(args)
    