#!/usr/bin/env python3
"""
Simple runner script for the Dual Agent Framework
Simplified interface for running tasks with Coordinator + Operator agents
"""

import argparse
import datetime
import json
import logging
import os
import sys
import shutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from functools import partial
from multiprocessing import Pool, cpu_count
from mm_agents.my_agent import MyAgentFramework
import traceback
import glob
from utils import build_additional_contexts

def process_single_task(
    domain: str,
    task_id: str,
    cfg: dict,
    **kwargs
) -> Tuple[str, float]:
    """Process a single task with the dual agent framework."""
    # Extract parameters
    path_to_vm = kwargs.get('path_to_vm')
    snapshot_name = kwargs.get('snapshot_name', 'init_state')
    result_dir = kwargs.get('result_dir', './results/dual_agent')
    coordinator_model = kwargs.get('coordinator_model', 'o3')
    llm_config_path = kwargs.get('llm_config_path', 'mm_agents/coact/OAI_CONFIG_LIST')
    max_steps = kwargs.get('max_steps', 15)
    screen_width = kwargs.get('screen_width', 1920)
    screen_height = kwargs.get('screen_height', 1080)
    sleep_after_execution = kwargs.get('sleep_after_execution', 0.5)
    client_password = kwargs.get('client_password', '')
    rag = kwargs.get('rag', False)
    rag_topk = kwargs.get('rag_topk', 4)
    rag_filename = kwargs.get('rag_filename', '')
    headless = kwargs.get('headless', False)
    verbose_instruction = kwargs.get('verbose_instruction', False)
    
    # Setup result directory
    history_save_dir = os.path.join(result_dir, f"{domain}/{task_id}")
    os.makedirs(history_save_dir, exist_ok=True)
    
    # Construct example path for verbose instruction
    test_config_base_dir = kwargs.get('test_config_base_dir', 'evaluation_examples/examples')
    example_path = os.path.join(test_config_base_dir, f"{domain}/{task_id}")
    
    # Build context using the common function
    additional_context = build_additional_contexts(
        task_config=cfg,
        example_dir=example_path,
        use_rag=rag,
        use_verbose_instruction=verbose_instruction,
        rag_topk=rag_topk,
        rag_filename=rag_filename
    )
    
    try:
        # Initialize framework
        framework = MyAgentFramework(
            coordinator_model=coordinator_model,
            operator_client_password=client_password,
            screen_width=screen_width,
            screen_height=screen_height,
            sleep_after_execution=sleep_after_execution,
            llm_config_path=llm_config_path,
            max_steps=max_steps,
            history_save_dir=history_save_dir
        )
        
        # Setup environment
        framework.setup_environment(
            path_to_vm=path_to_vm,
            snapshot_name=snapshot_name,
            headless=headless
        )
        
        # Execute task
        score, chat_history = framework.execute_task(kwargs, cfg, additional_context)
        
        # Save results
        with open(os.path.join(history_save_dir, "result.txt"), "w") as f:
            f.write(str(score))
        
        # Cleanup
        framework.cleanup()
        
        # Read execution log for statistics
        execution_log_path = os.path.join(history_save_dir, "execution_log.json")
        if os.path.exists(execution_log_path):
            with open(execution_log_path, "r") as f:
                execution_log = json.load(f)
                stats = execution_log.get("statistics", {})
                gui_ops = stats.get("gui_operations", 0)
                code_ops = stats.get("code_executions", 0)
                total_cost = stats.get("total_cost", 0)
                
                print(f"Task {domain}/{task_id} completed with score: {score}")
                print(f"Total operations: {gui_ops + code_ops} (GUI: {gui_ops}, Code: {code_ops})")
                print(f"Total cost: ${total_cost:.4f}")
        else:
            print(f"Task {domain}/{task_id} completed with score: {score}")
        return domain, score
        
    except Exception as e:
        print(f"Error processing task {domain}/{task_id}")
        traceback.print_exc()
        score = 0.0
        
        # Save error information
        with open(os.path.join(history_save_dir, "result.txt"), "w") as f:
            f.write(str(score))
        with open(os.path.join(history_save_dir, "err_reason.txt"), "w") as f:
            f.write(f"Fatal error: {str(e)}")
        
        return domain, 0.0

def main():
    parser = argparse.ArgumentParser(description="Run dual agent framework evaluation")
    
    # Environment config
    parser.add_argument("--path_to_vm", type=str, default="vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx",
                       help="Path to VM file")
    parser.add_argument("--snapshot_name", type=str, default="init_state")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument("--client_password", type=str, default="password",
                       help="VM client password")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    # Agent config
    parser.add_argument("--llm_config_path", type=str, default="mm_agents/coact/OAI_CONFIG_LIST")
    parser.add_argument("--coordinator_model", type=str, default="o3",
                       help="Model for Coordinator agent")
    parser.add_argument("--max_steps", type=int, default=15,
                       help="Maximum steps for Coordinator")

    # Task config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default=os.path.join('evaluation_examples', 'test_one.json'))
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples/examples")
    parser.add_argument("--rerun", action="store_true", help="Rerun tests that have already been run")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores")

    # RAG config
    parser.add_argument("--rag", action='store_true', help="Enable RAG context")
    parser.add_argument("--rag_topk", type=int, default=4)
    
    # RAG config
    parser.add_argument("--rag_filename", type=str, default="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt")

    # Verbose instruction config
    parser.add_argument("--verbose_instruction", action='store_true', help="Enable verbose instruction loading")

    # Output config
    parser.add_argument("--result_dir", type=str, default="./results/dual_agent",
                       help="Directory to save results")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")
    
    args = parser.parse_args()
    
    # Setup logging configuration
    result_name = os.path.basename(args.result_dir)
    logger = logging.getLogger()

    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

    datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join("logs", "{:}-error-{:}.log".format(result_name, datetime_str)), encoding="utf-8"
    )
    debug_handler = logging.FileHandler(
        os.path.join("logs", "{:}-debug-{:}.log".format(result_name, datetime_str)), encoding="utf-8"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(logging.ERROR)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
    )
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    stdout_handler.addFilter(logging.Filter("desktopenv"))

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    #  }}} Logger Configs #

    logger = logging.getLogger("desktopenv.expeiment")

    # Load test metadata
    with open(args.test_all_meta_path, encoding="utf-8") as f:
        test_all_meta = json.load(f)
    
    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}
    
    if not args.get_score:
        tasks = []
        scores: Dict[str, List[float]] = {}
        
        # Process each domain and example
        for domain in test_all_meta:
            scores[domain] = []
            for task_id in test_all_meta[domain]:
                target_dir = os.path.join(args.result_dir, f"{domain}/{task_id}")
                result_path = os.path.join(target_dir, 'result.txt')
                cfg = json.load(open(os.path.join(args.test_config_base_dir, f"{domain}/{task_id}/{task_id}.json")))
                
                # Check if we should skip this task
                should_skip = False
                if not args.rerun and os.path.exists(result_path) and not os.path.exists(os.path.join(target_dir, 'error.txt')):
                    result = float(open(result_path, 'r').read())
                    print(f"Results already exist in {domain}/{task_id}, result: {result}")
                    
                    # Skip successful tasks, or skip failed tasks if not rerun_fail
                    if result > 0.0 or not args.rerun_fail:
                        should_skip = True
                
                if not should_skip:
                    # Clean up existing directory and add to tasks
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                    tasks.append((domain, task_id, cfg))

        # Check if there are any tasks to process
        if not tasks:
            print("No tasks to process. All tasks have already been completed.")
            # Print summary of existing results
            print("\n=== Summary of Existing Results ===")
            for domain in test_all_meta:
                domain_scores = []
                for ex_id in test_all_meta[domain]:
                    score_file = os.path.join(args.result_dir, f"{domain}/{ex_id}/result.txt")
                    if os.path.exists(score_file):
                        with open(score_file, "r") as f:
                            domain_scores.append(float(f.read()))
                if domain_scores:
                    avg_score = sum(domain_scores) / len(domain_scores)
                    print(f"{domain}: {len(domain_scores)} tasks, average score: {avg_score:.2%}")
        else:
            # Use multiprocessing to process tasks in parallel
            num_workers = min(cpu_count() // 2, args.num_envs)
            print(f"Processing {len(tasks)} tasks with {num_workers} workers...")

            # Create a partial function with fixed parameters
            task_kwargs = {
                'path_to_vm': args.path_to_vm,
                'snapshot_name': args.snapshot_name,
                'result_dir': args.result_dir,
                'coordinator_model': args.coordinator_model,
                'llm_config_path': args.llm_config_path,
                'max_steps': args.max_steps,
                'screen_width': args.screen_width,
                'screen_height': args.screen_height,
                'sleep_after_execution': args.sleep_after_execution,
                'client_password': args.client_password,
                'rag': args.rag,
                'rag_topk': args.rag_topk,
                'rag_filename': args.rag_filename,
                'headless': args.headless,
                'test_config_base_dir': args.test_config_base_dir,
                'verbose_instruction': args.verbose_instruction
            }

            if args.num_envs > 1:
                # Parallel processing
                with Pool(processes=num_workers) as pool:
                    process_func = partial(process_single_task, **task_kwargs)
                    results = pool.starmap(process_func, tasks)
            else:
                # Sequential processing
                results = []
                for domain, task_id, cfg in tasks:
                    result = process_single_task(domain, task_id, cfg, **task_kwargs)
                    results.append(result)

            # Collect scores from results
            for domain, score in results:
                scores[domain].append(score)

    # Calculate and display final results with cost information
    all_scores = []
    all_costs = []
    count_remain = 0
    scores = {}
    costs = {}
    operations_stats = {}
    
    for domain in test_all_meta:
        scores[domain] = []
        costs[domain] = []
        operations_stats[domain] = {"gui_ops": [], "code_ops": [], "total_ops": []}
        
        for ex_id in test_all_meta[domain]:
            score_file = os.path.join(args.result_dir, f"{domain}/{ex_id}/result.txt")
            execution_log_file = os.path.join(args.result_dir, f"{domain}/{ex_id}/execution_log.json")
            
            if os.path.exists(score_file):
                # Read score
                with open(score_file, "r") as f:
                    score = eval(f.read())
                    all_scores.append(score)
                    scores[domain].append(score)
                
                # Read cost and operations info from execution log
                cost = 0.0
                gui_ops = 0
                code_ops = 0
                
                if os.path.exists(execution_log_file):
                    try:
                        with open(execution_log_file, "r") as f:
                            execution_log = json.load(f)
                            stats = execution_log.get("statistics", {})
                            cost = stats.get("total_cost", 0.0)
                            gui_ops = stats.get("gui_operations", 0)
                            code_ops = stats.get("code_executions", 0)
                    except Exception as e:
                        print(f"Warning: Could not read execution log for {domain}/{ex_id}: {e}")
                
                all_costs.append(cost)
                costs[domain].append(cost)
                operations_stats[domain]["gui_ops"].append(gui_ops)
                operations_stats[domain]["code_ops"].append(code_ops)
                operations_stats[domain]["total_ops"].append(gui_ops + code_ops)
            else:
                all_scores.append(0.0)
                all_costs.append(0.0)
                scores[domain].append(0.0)
                costs[domain].append(0.0)
                operations_stats[domain]["gui_ops"].append(0)
                operations_stats[domain]["code_ops"].append(0)
                operations_stats[domain]["total_ops"].append(0)
                count_remain += 1
    
    print('=== Overall Results ===')
    for domain in scores:
        if scores[domain]:
            avg_score = sum(scores[domain]) / len(scores[domain])
            total_cost = sum(costs[domain])
            avg_cost = total_cost / len(costs[domain]) if costs[domain] else 0
            print(f"{domain}: {len(scores[domain])} tasks, average score: {avg_score:.2%}")
    
    all_avg_score = np.mean(all_scores)
    total_cost = sum(all_costs)
    avg_cost = total_cost / len(all_costs) if all_costs else 0
    
    # Calculate total operations
    total_gui_ops = sum(sum(operations_stats[domain]["gui_ops"]) for domain in operations_stats)
    total_code_ops = sum(sum(operations_stats[domain]["code_ops"]) for domain in operations_stats)
    total_ops = total_gui_ops + total_code_ops
    tasks_processed = len(all_scores) - count_remain
    avg_ops_per_task = total_ops / tasks_processed if tasks_processed > 0 else 0
    avg_gui_ops_per_task = total_gui_ops / tasks_processed if tasks_processed > 0 else 0
    avg_code_ops_per_task = total_code_ops / tasks_processed if tasks_processed > 0 else 0
    
    print(f"Average score: {all_avg_score:.2%}")
    print(f"Total cost: ${total_cost:.4f} (average per task: ${avg_cost:.4f})")
    print(f"Total operations: {total_ops} (average per task: {avg_ops_per_task:.1f})")
    print(f"  - GUI operations: {total_gui_ops} (average per task: {avg_gui_ops_per_task:.1f})")
    print(f"  - Code executions: {total_code_ops} (average per task: {avg_code_ops_per_task:.1f})")
    print(f"Tasks processed: {tasks_processed}, tasks remain: {count_remain}")


if __name__ == "__main__":
    main()