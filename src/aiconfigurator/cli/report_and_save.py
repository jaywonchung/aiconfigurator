# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import random
import traceback
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from prettytable import PrettyTable

from aiconfigurator.generator.api import generate_backend_config
from aiconfigurator.generator.cli_args import build_dynamo_config
from aiconfigurator.sdk import pareto_analysis
from aiconfigurator.sdk.pareto_analysis import draw_pareto_to_string
from aiconfigurator.sdk.task import TaskConfig, task_config_to_generator_config
from aiconfigurator.sdk.utils import safe_mkdir

logger = logging.getLogger(__name__)


def _plot_worker_setup_table(exp_name: str, config_df: pd.DataFrame, total_gpus: int, tpot_target: float, top: int, is_moe: bool) -> str:
    """Plot worker setup table for a single experiment."""
    buf = []

    if config_df is None or config_df.empty:
        return ""

    config_df['tokens/s/gpu_cluster'] = config_df['tokens/s/gpu'] * (total_gpus // config_df['num_total_gpus']) \
        * config_df['num_total_gpus'] / total_gpus if total_gpus > 0 else 0
    # Show all configs instead of just top N
    top_configs = config_df[config_df['tpot'] <= tpot_target].sort_values(by='tokens/s/gpu_cluster', ascending=False).copy()

    if top_configs.empty:
        return f"\nNo configurations for {exp_name} met the TPOT constraint."

    top_configs['replicas'] = total_gpus // top_configs['num_total_gpus']
    top_configs['total_gpus_used'] = top_configs['num_total_gpus'] * top_configs['replicas']

    buf.append(f"\n{exp_name} Pareto-Optimal Configurations: (Sorted by tokens/s/gpu)")
    table = PrettyTable()
    
    # Check if it is disagg config by checking for prefill/decode specific columns
    is_disagg = '(p)tp' in top_configs.columns

    if is_disagg:
        table.field_names = ["Rank", f"\033[1mtokens/s/gpu\033[0m", "tokens/s/user", "TTFT", "TPOT", "concurrency", "total_gpus(used)", "replicas", "gpus/replica",
                             "(p)workers", "(p)gpus/worker", "(p)parallel", "(p)bs", "(p)power_limit", "(p)power/GPU",
                             "(d)workers", "(d)gpus/worker", "(d)parallel", "(d)bs", "(d)power_limit", "(d)power/GPU",
                             "cluster_power"]
        for i, row in enumerate(top_configs.to_dict('records')):
            if is_moe:
                p_parallel = f'tp\033[4m{row["(p)tp"]}\033[0mpp\033[4m{row["(p)pp"]}\033[0mdp\033[4m{row["(p)dp"]}\033[0metp{row["(p)moe_tp"]}ep{row["(p)moe_ep"]}'
                d_parallel = f'tp\033[4m{row["(d)tp"]}\033[0mpp\033[4m{row["(d)pp"]}\033[0mdp\033[4m{row["(d)dp"]}\033[0metp{row["(d)moe_tp"]}ep{row["(d)moe_ep"]}'
                p_gpus_worker = f'{row["(p)pp"]*row["(p)tp"]*row["(p)dp"]} (=\033[4m{row["(p)tp"]}\033[0mx\033[4m{row["(p)pp"]}\033[0mx\033[4m{row["(p)dp"]}\033[0m)'
                d_gpus_worker = f'{row["(d)pp"]*row["(d)tp"]*row["(d)dp"]} (=\033[4m{row["(d)tp"]}\033[0mx\033[4m{row["(d)pp"]}\033[0mx\033[4m{row["(d)dp"]}\033[0m)'
            else:
                p_parallel = f'tp\033[4m{row["(p)tp"]}\033[0mpp\033[4m{row["(p)pp"]}\033[0m'
                d_parallel = f'tp\033[4m{row["(d)tp"]}\033[0mpp\033[4m{row["(d)pp"]}\033[0m'
                p_gpus_worker = f'{row["(p)pp"]*row["(p)tp"]} (=\033[4m{row["(p)tp"]}\033[0mx\033[4m{row["(p)pp"]}\033[0m)'
                d_gpus_worker = f'{row["(d)pp"]*row["(d)tp"]} (=\033[4m{row["(d)tp"]}\033[0mx\033[4m{row["(d)pp"]}\033[0m)'

            p_power_limit = f"{int(row['(p)power_limit'])}W" if '(p)power_limit' in row and not pd.isna(row['(p)power_limit']) else "N/A"
            p_power = f"{row['(p)power']:.1f}W" if '(p)power' in row and row['(p)power'] > 0 else "N/A"
            d_power_limit = f"{int(row['(d)power_limit'])}W" if '(d)power_limit' in row and not pd.isna(row['(d)power_limit']) else "N/A"
            d_power = f"{row['(d)power']:.1f}W" if '(d)power' in row and row['(d)power'] > 0 else "N/A"
            cluster_power = f"{row['total_cluster_power'] * row['replicas']:.1f}W" if 'total_cluster_power' in row and row['total_cluster_power'] > 0 else "N/A"

            table.add_row([
                i + 1, f"\033[1m{row['tokens/s/gpu_cluster']:.2f}\033[0m", f"{row['tokens/s/user']:.2f}", f"{row['ttft']:.2f}", f"{row['tpot']:.2f}",
                f"{row['concurrency']*row['replicas']}(={row['concurrency']}x{row['replicas']})",
                f"{total_gpus} ({row['total_gpus_used']}={row['replicas']}x{row['num_total_gpus']})", row['replicas'],
                f"{row['num_total_gpus']} (={row['(p)workers']}x{row['(p)pp']*row['(p)tp']*row['(p)dp']}+{row['(d)workers']}x{row['(d)pp']*row['(d)tp']*row['(d)dp']})",
                row['(p)workers'], p_gpus_worker, p_parallel, row['(p)bs'], p_power_limit, p_power,
                row['(d)workers'], d_gpus_worker, d_parallel, row['(d)bs'], d_power_limit, d_power,
                cluster_power,
            ])
    else: # agg
        table.field_names = ["Rank", f"\033[1mtokens/s/gpu\033[0m", "tokens/s/user", "TTFT", "TPOT", "concurrency", "total_gpus(used)",
                             "replicas", "gpus/replica", "gpus/worker", "parallel", "bs", "power_limit", "power/GPU", "cluster_power"]
        for i, row in enumerate(top_configs.to_dict('records')):
            if is_moe:
                parallel = f'tp\033[4m{row["tp"]}\033[0mpp\033[4m{row["pp"]}\033[0mdp\033[4m{row["dp"]}\033[0metp{row["moe_tp"]}ep{row["moe_ep"]}'
                gpus_worker = f'{row["pp"]*row["tp"]*row["dp"]} (=\033[4m{row["tp"]}\033[0mx\033[4m{row["pp"]}\033[0mx\033[4m{row["dp"]}\033[0m)'
            else:
                parallel = f'tp\033[4m{row["tp"]}\033[0mpp\033[4m{row["pp"]}\033[0m'
                gpus_worker = f'{row["pp"]*row["tp"]} (=\033[4m{row["tp"]}\033[0mx\033[4m{row["pp"]}\033[0m)'

            power_limit = f"{int(row['power_limit'])}W" if 'power_limit' in row and not pd.isna(row['power_limit']) else "N/A"
            per_gpu_power = f"{row['power']:.1f}W" if 'power' in row and row['power'] > 0 else "N/A"
            cluster_power = f"{row['total_cluster_power'] * row['replicas']:.1f}W" if 'total_cluster_power' in row and row['total_cluster_power'] > 0 else "N/A"

            table.add_row([
                i + 1, f"\033[1m{row['tokens/s/gpu_cluster']:.2f}\033[0m", f"{row['tokens/s/user']:.2f}", f"{row['ttft']:.2f}", f"{row['tpot']:.2f}",
                f"{row['concurrency']*row['replicas']}(={row['concurrency']}x{row['replicas']})", f"{total_gpus} ({row['total_gpus_used']}={row['replicas']}x{row['num_total_gpus']})",
                row['replicas'], row['num_total_gpus'],
                gpus_worker, parallel, row['bs'], power_limit, per_gpu_power, cluster_power
            ])
            
    buf.append(table.get_string())
    return "\n".join(buf)
    
def log_final_summary(
        chosen_exp: str, 
        best_throughputs: Dict[str, float], 
        best_configs: Dict[str, pd.DataFrame], 
        pareto_fronts: Dict[str, pd.DataFrame], 
        task_configs: Dict[str, TaskConfig],
        mode: str,
):
    """Log final summary of configuration results"""
    
    # Consolidate and format results into a summary box for clear presentation
    summary_box = []
    summary_box.append("*" * 80)
    summary_box.append("*{:^78}*".format(" Dynamo aiconfigurator Final Results "))
    summary_box.append("*" * 80)

    summary_box.append("  " + "-" * 76)
    summary_box.append("  Input Configuration & SLA Target:")
    summary_box.append(f"    Model: {task_configs[chosen_exp].config.model_name} (is_moe: {task_configs[chosen_exp].config.is_moe})")
    summary_box.append(f"    Total GPUs: {task_configs[chosen_exp].total_gpus}")
    if mode == "default":
        agg_value = best_throughputs.get("agg", 0.0)
        disagg_value = best_throughputs.get("disagg", 0.0)
        if agg_value > 0 and disagg_value > 0:
            benefit_ratio = disagg_value / agg_value
        elif agg_value == 0 and disagg_value > 0:
            benefit_ratio = float("inf")
        elif agg_value > 0 and disagg_value == 0:
            benefit_ratio = 0.0
        else:
            benefit_ratio = 0.0 # handle case where both are 0
        summary_box.append(f"    Best Experiment Chosen: \033[1m{chosen_exp} at {best_throughputs[chosen_exp]:.2f} tokens/s/gpu (disagg {benefit_ratio:.2f}x better)\033[0m")        
    else:
        summary_box.append(f"    Best Experiment Chosen: \033[1m{chosen_exp} at {best_throughputs[chosen_exp]:.2f} tokens/s/gpu\033[0m")
        
    summary_box.append("  " + "-" * 76)


    # ============================= overall summary
    summary_box.append("  Overall Best Configuration:")
    best_config_df = best_configs[chosen_exp]
    best_throughput = best_throughputs[chosen_exp]
    
    summary_box.append(f"    - Best Throughput: {best_throughput:.2f} tokens/s/gpu")
    if not best_config_df.empty:
        best_conf_details = best_config_df.iloc[0]
        summary_box.append(f"    - User Throughput: {best_conf_details['tokens/s/user']:.2f} tokens/s/user")
        summary_box.append(f"    - TTFT: {best_conf_details['ttft']:.2f}ms")
        summary_box.append(f"    - TPOT: {best_conf_details['tpot']:.2f}ms")

        # Display power information if available
        if 'total_cluster_power' in best_conf_details and best_conf_details['total_cluster_power'] > 0:
            summary_box.append(f"    - Total Cluster Power: {best_conf_details['total_cluster_power']:.1f}W")
            if 'power' in best_conf_details and best_conf_details['power'] > 0:
                summary_box.append(f"    - Per-GPU Power: {best_conf_details['power']:.1f}W")

        # Display power budget status if applicable
        if 'within_power_budget' in best_conf_details:
            budget_status = "✓ Within" if best_conf_details['within_power_budget'] else "✗ Over"
            summary_box.append(f"    - Power Budget Status: {budget_status}")
    summary_box.append("  " + "-" * 76)

    # ============================= pareto frontier
    pareto_plot_buf = ""
    if len(pareto_fronts) <= 10:  # avoid overly crowded plots
        summary_box.append("  Pareto Frontier:")

        # Display power budget if set
        cluster_power_budget = getattr(task_configs[chosen_exp], 'cluster_power_budget', None)
        if cluster_power_budget is not None:
            summary_box.append(f"    Power Budget Constraint: {cluster_power_budget}W total cluster power")

        series_payload = []

        # Build series, splitting by power budget if applicable
        for name, df in pareto_fronts.items():
            if df is None or df.empty:
                continue

            # Check if power budget filtering is enabled
            if 'within_power_budget' in df.columns and df['within_power_budget'].notna().any():
                # Split into within budget (green) and over budget (red)
                within_df = df[df['within_power_budget'] == True]
                over_df = df[df['within_power_budget'] == False]

                if not within_df.empty:
                    series_payload.append({
                        "df": within_df,
                        "label": f"{name} (within budget)",
                        "color": (144, 238, 144),  # light green
                    })
                if not over_df.empty:
                    series_payload.append({
                        "df": over_df,
                        "label": f"{name} (over budget)",
                        "color": (255, 99, 71),  # tomato red
                    })
            else:
                # No power budget - plot normally
                series_payload.append({"df": df, "label": name})

        highlight_series = None
        if not best_config_df.empty:
            highlight_series = {
                "df": best_config_df.head(1),
                "label": f"{chosen_exp} best",
            }
        pareto_plot_buf = draw_pareto_to_string(
            f"{task_configs[chosen_exp].config.model_name} Pareto Frontier",
            series_payload,
            highlight=highlight_series,
        )
        summary_box.append(pareto_plot_buf)
    summary_box.append("  " + "-" * 76)

    # ============================= deployment details
    summary_box.append("  Deployment Details:")
    summary_box.append(f"    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system")
    summary_box.append(f"    Some math: total gpus used = replicas * gpus/replica")
    summary_box.append(f"               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker")
    summary_box.append(f"               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined \033[4mnumbers\033[0m are the actual values in math)")
    
    # Plot worker setup tables for all experiments using Pareto frontier
    for exp_name, pareto_df in pareto_fronts.items():
        if pareto_df is None or pareto_df.empty:
            continue
        exp_task_config = task_configs[exp_name].config
        total_gpus = getattr(task_configs[exp_name], "total_gpus", None) or 0
        table_buf = _plot_worker_setup_table(exp_name, pareto_df, total_gpus, exp_task_config.runtime_config.tpot, 5, exp_task_config.is_moe)
        summary_box.append(table_buf)

    summary_box.append("*" * 80)
    logger.info("\n" + "\n".join(summary_box))

def save_results(
    args,
    best_configs: Dict[str, pd.DataFrame], 
    pareto_fronts: Dict[str, pd.DataFrame], 
    task_configs: Dict[str, TaskConfig], 
    save_dir: str,
    generated_backend_version: Optional[str] = None,
):
    """Save the results to a directory."""
    
    first_exp_name = list(task_configs.keys())[0]
    first_task_config = task_configs[first_exp_name].config
    
    result_prefix = f"{first_task_config.model_name}_isl{first_task_config.runtime_config.isl}_osl{first_task_config.runtime_config.osl}_ttft{int(first_task_config.runtime_config.ttft)}_tpot{int(first_task_config.runtime_config.tpot)}"
    result_dir_path = os.path.join(save_dir, f'{result_prefix}_{random.randint(0,1000000)}')
    
    logger.info(f'Saving results to {result_dir_path}')
    try:
        safe_result_dir = safe_mkdir(result_dir_path, exist_ok=True)

        # Save overall pareto plots in the root directory
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plt.title(f"{first_task_config.model_name} tokens/s/gpu vs tokens/s/user")
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        for i, (exp_name, pareto_df) in enumerate(pareto_fronts.items()):
            if not pareto_df.empty:
                pareto_analysis.draw_pareto(
                    pareto_df, 'tokens/s/user', 'tokens/s/gpu', ax, colors[i % len(colors)], exp_name
                )
        plt.savefig(os.path.join(safe_result_dir, 'pareto_frontier.png'))
        plt.close()

        # Save each experiment's results in its own subdirectory
        for exp_name, pareto_df in pareto_fronts.items():
            exp_dir = os.path.join(safe_result_dir, exp_name)
            safe_mkdir(exp_dir, exist_ok=True)

            # 1. Save best config dataframe
            best_config_df = best_configs.get(exp_name) # top n configs
            if best_config_df is not None:
                best_config_df.to_csv(os.path.join(exp_dir, 'best_config_topn.csv'), index=False)

            # 2. Save all pareto dataframe
            if pareto_df is not None:
                pareto_df.to_csv(os.path.join(exp_dir, 'pareto.csv'), index=False)

            # 3. Save the config for this experiment
            exp_task_config = task_configs[exp_name]

            with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f: # for future aic repro
                yaml.safe_dump(json.loads(exp_task_config.pretty()), f, sort_keys=False)
            
            # 4. Save the generated config for this experiment, sub-directory for each best config
            if best_config_df is not None:
                dynamo_overrides = build_dynamo_config(args)
                for i, (idx, result_df) in enumerate(best_config_df.iterrows()):
                    cfg = task_config_to_generator_config(task_config=exp_task_config, result_df=result_df)

                    top_config_dir = os.path.join(exp_dir, f'top{i+1}')
                    safe_mkdir(top_config_dir, exist_ok=True)
                    with open(os.path.join(top_config_dir, 'generator_config.yaml'), 'w') as f:
                        yaml.safe_dump(cfg, f, sort_keys=False)
                    
                    try:
                        artifacts = generate_backend_config.from_runtime(
                            cfg=cfg,
                            backend=exp_task_config.backend_name,
                            version=generated_backend_version or exp_task_config.backend_version,
                            overrides=dynamo_overrides,                    
                            save_dir=top_config_dir,
                        )
                    except Exception as exc:
                        logger.warning("Failed to generate backend config from aic generator: %s, %s", exc, traceback.format_exc())

    except Exception as exc:
        logger.error("Failed to save results: %s, %s", exc, traceback.format_exc())
