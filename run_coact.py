import argparse
import base64
import glob
import datetime
import shutil
import traceback
from typing import Dict, List
import json
import time
import os
from mm_agents.coact.operator_agent import OrchestratorAgent, OrchestratorUserProxyAgent
from mm_agents.coact.autogen import LLMConfig
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
from configs.config import OPENAI_API_KEY
from pydantic import SecretStr
import numpy as np

TASK_DESCRIPTION = """# Your role
You are a task solver, you need to complete a computer-using task step-by-step.
1. Describe the screenshot.
2. Provide a detailed plan, including a list of user requirements like specific file name, file path, etc.
3. Follow the following instructions and complete the task with your skills.
    - If you think the task is impossible to complete (no file, wrong environment, etc.), reply with "INFEASIBLE" to end the conversation.
    - **Do not** do (or let coding/GUI agent do) anything else out of the user's instruction like change the file name. This will make the task fail.
    - Check every screenshot carefully and see if it fulfills the task requirement.
    - You MUST try the Coding Agent first for file operation tasks like spreadsheet modification.
4. Verify the result and see if it fulfills the user's requirement.

# Your helpers
You can use the following tools to solve the task. You can only call one of gui agent or coding agent per reply:

## Programmer
Let a programmer to solve a subtask you assigned. 
The Programmer can write python or bash code to modify almost everything in the computer, like files, apps, system settings, etc. 
It requires a environment description and a detailed task description. As detailed as possible.
Can use any python package you instructed.
Will return a summary with the output of the code.
When letting coding agent to modify the spreadsheet, after the task completed, you MUST make sure EVERY modified value in the spreadsheet is in the desired position (e.g., filled in the expected cell) by a GUI Operator.
After that, if anything is wrong, tell the programmer to modify it.

## GUI Operator
Let a GUI agent to solve a subtask you assigned. 
GUI agent can operate the computer by clicking and typing (but not accurate). 
Require a detailed task description.
When you call GUI agent, it will only have a **20-step** budget to complete your task. Each step is a one-time interaction with OS like mouse click or keyboard typing. Please take this into account when you plan the actions.
If you let GUI Operator to check the result, you MUST let it close and reopen the file because programmer's result will NOT be updated to the screen. 
"""


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default="vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx")
    parser.add_argument("--snapshot_name", type=str, default="init_state")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument("--client_password", type=str, default="password") # osworld-public-evaluation for aws

    # agent config
    parser.add_argument("--oai_config_path", type=str, default="mm_agents/coact/OAI_CONFIG_LIST")
    parser.add_argument("--orchestrator_model", type=str, default="o3")
    parser.add_argument("--coding_model", type=str, default="o4-mini")
    parser.add_argument("--cua_model", type=str, default="computer-use-preview")
    parser.add_argument("--orchestrator_max_steps", type=int, default=15) #15
    parser.add_argument("--coding_max_steps", type=int, default=10) #20
    parser.add_argument("--cua_max_steps", type=int, default=10) #25
    parser.add_argument("--cut_off_steps", type=int, default=20) #200

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default=os.path.join('evaluation_examples', 'test_one.json')
    )
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples/examples"
    )
    parser.add_argument("--rerun", action="store_true", help="Rerun tests that have already been run")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores")

    # RAG related
    parser.add_argument("--rag", action='store_true', help="Whether to use RAG for the agent")
    parser.add_argument("--rag_topk", type=int, default=4, help="Top k to use for RAG")
    parser.add_argument("--rag_filename", type=str, default="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt", help="RAG retrieved context file name")

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results/coact_15_10_10_20")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")

    args = parser.parse_args()
    return args

args = config()

logger = logging.getLogger()

log_level = getattr(logging, args.log_level.upper())
logger.setLevel(log_level)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
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


def get_retrieved_context(config_path: str, topk: int = 4, file_name: str = "retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt") -> str:
    """Get retrieved context from RAG file"""
    context_path = os.path.join(os.path.dirname(config_path), file_name)
    if os.path.exists(context_path):
        with open(context_path, "r", encoding="utf-8") as f:
            context = f.read().strip()
        if context.strip() == "": return None
        splits = context.split("Documentation Source:")
        if len(splits) > topk + 1: # the first is ""
            return "Documentation Source:".join(splits[:topk + 1])
        return context
    raise ValueError(f"Retrieved context not found under {os.path.dirname(config_path)}")

def save_args_to_settings(args, result_dir):
    """Save args to settings.txt in the result subdirectory"""
    os.makedirs(result_dir, exist_ok=True)
    settings_file = os.path.join(result_dir, "settings.txt")
    
    with open(settings_file, "w", encoding="utf-8") as f:
        args_dict = vars(args)
        for key, value in sorted(args_dict.items()):
            f.write(f"{key} = {value}\n")

def process_task(task_info, 
                path_to_vm,
                snapshot_name="init_state",
                orchestrator_model="o3",
                coding_model='o4-mini',
                result_dir='results/coact',
                orchestrator_max_steps=15,
                cua_max_steps=25,
                coding_max_steps=20,
                cut_off_steps=150,
                screen_width=1920,
                screen_height=1080,
                sleep_after_execution=0.5,
                config_path="OAI_CONFIG_LIST",
                client_password="",
                rag=False,
                rag_topk=4,
                rag_filename="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt",
                ):
    """Worker function to process a single task"""
    domain, ex_id, cfg = task_info
    
    # Recreate llm_config inside the worker process
    llm_config = LLMConfig.from_json(path=config_path).where(model=orchestrator_model)
    
    # Update all api_key values to use OPENAI_API_KEY from config
    for config_item in llm_config.config_list:
        config_item.api_key = SecretStr(OPENAI_API_KEY)
    
    history_save_dir = os.path.join(result_dir, f"{domain}/{ex_id}")
    if not os.path.exists(history_save_dir):
        os.makedirs(history_save_dir)
    
    task_config = json.load(open(cfg))

    # Add RAG context if enabled
    if rag: 
        task_config['context'] = get_retrieved_context(cfg, rag_topk, file_name=rag_filename)
    else: 
        task_config['context'] = None

    try:
        with llm_config:
            orchestrator = OrchestratorAgent(
                name="orchestrator",
                system_message=TASK_DESCRIPTION
            )
            orchestrator_proxy = OrchestratorUserProxyAgent(
                name="orchestrator_proxy",
                is_termination_msg=lambda x: x.get("content", "") and (x.get("content", "")[0]["text"].lower() == "terminate" or x.get("content", "")[0]["text"].lower() == "infeasible"),
                human_input_mode="NEVER",
                path_to_vm=path_to_vm,
                snapshot_name=snapshot_name,
                screen_width=screen_width,
                screen_height=screen_height,
                sleep_after_execution=sleep_after_execution,
                code_execution_config=False,
                history_save_dir=history_save_dir,
                llm_model=coding_model,
                truncate_history_inputs=cua_max_steps + 1,
                cua_max_steps=cua_max_steps,
                coding_max_steps=coding_max_steps,
                cut_off_steps=cut_off_steps,
                client_password=client_password,
                user_instruction=task_config["instruction"]
            )

        orchestrator_proxy.reset(task_config=task_config)
        time.sleep(60)
        screenshot = orchestrator_proxy.env.controller.get_screenshot()

        with open(os.path.join(history_save_dir, f'initial_screenshot_orchestrator.png'), "wb") as f:
            f.write(screenshot)
            
        # Prepare the initial message with optional RAG context
        initial_message = f"""{task_config["instruction"]}
Check my computer screenshot and describe it first. If this task is possible to complete, please complete it on my computer. If not, reply with "INFEASIBLE" to end the conversation.
I will not provide further information to you."""

        # Add RAG context if available
        if task_config.get('context'):
            context_message = f"\n\nWe also retrieve relevant documentation from the web to help you with the task:\n{task_config['context']}"
            initial_message += context_message

        initial_message += "<img data:image/png;base64," + base64.b64encode(screenshot).decode("utf-8") + ">"
        
        orchestrator_proxy.initiate_chat(
            recipient=orchestrator,
            message=initial_message,
            max_turns=orchestrator_max_steps
        )
        
        chat_history = []
        key = list(orchestrator_proxy.chat_messages.keys())[0]
        chat_messages = orchestrator_proxy.chat_messages[key]
        for item in chat_messages:
            item.pop('tool_responses', None)
            if item.get('role', None) in ['tool', 'assistant'] and item.get('content', None):
                for msg in item['content']:
                    if msg.get('type', None) == 'image_url':
                        msg['image_url'] = "<image>"
            chat_history.append(item)
        
        with open(os.path.join(history_save_dir, f'chat_history.json'), "w") as f:
            json.dump(chat_history, f)

        if chat_history[-1]['role'] == 'user' and 'INFEASIBLE' in chat_history[-1]['content'][0]['text']:
            orchestrator_proxy.env.action_history.append("FAIL")

        cua_steps = len(glob.glob(f"{history_save_dir}/cua_output*/step_*.png"))
        coding_paths = glob.glob(f"{history_save_dir}/coding_output*/chat_history.json")
        coding_steps = 0
        for hist in coding_paths:
            with open(hist, 'r') as f:
                hist = json.dumps(json.load(f))
                coding_steps += hist.count('exitcode:')
        # 更新：cut_off_steps已经更新到orchestrator_proxy
        # if cua_steps + coding_steps > cut_off_steps:
        #     score = 0.0
        # else:
        #     score = orchestrator_proxy.env.evaluate()
        score = orchestrator_proxy.env.evaluate()
        print(f"Score: {score}")
        
        with open(os.path.join(history_save_dir, f'result.txt'), "w") as f:
            f.write(str(score))
        
        if orchestrator_proxy.env is not None:
            orchestrator_proxy.env.close()
            # orchestrator_proxy.env.clean_lock("./vmware_vm_data")
                
    except Exception as e:
        print(f"Error processing task {domain}/{ex_id}")
        traceback.print_exc()
        score = 0.0
        with open(os.path.join(history_save_dir, f'result.txt'), "w") as f:
            f.write(str(score))
        with open(os.path.join(history_save_dir, f'err_reason.txt'), "w") as f:
            f.write(f"Fatal error: {str(e)}")
    
    return domain, score


if __name__ == "__main__":
    args = config()

    with open(args.test_all_meta_path, encoding="utf-8") as f:
        test_all_meta = json.load(f)
    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}
    
    if not args.get_score:
        tasks = []
        scores: Dict[str, List[float]] = {}
        
        # Save args to settings.txt
        save_args_to_settings(args, args.result_dir)
        
        # Process each domain and example
        for domain in test_all_meta:
            scores[domain] = []
            for ex_id in test_all_meta[domain]:
                target_dir = os.path.join(args.result_dir, f"{domain}/{ex_id}")
                result_path = os.path.join(target_dir, 'result.txt')
                cfg = os.path.join(args.test_config_base_dir, f"{domain}/{ex_id}/{ex_id}.json")
                
                # Check if we should skip this task
                should_skip = False
                if not args.rerun and os.path.exists(result_path) and not os.path.exists(os.path.join(target_dir, 'err_reason.txt')):
                    result = float(open(result_path, 'r').read())
                    print(f"Results already exist in {domain}/{ex_id}, result: {result}")
                    
                    # Skip successful tasks, or skip failed tasks if not rerun_fail
                    if result > 0.0 or not args.rerun_fail:
                        should_skip = True
                
                if not should_skip:
                    # Clean up existing directory and add to tasks
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                    tasks.append((domain, ex_id, cfg))

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
            # Determine number of workers (you can adjust this based on your system)
            num_workers = min(cpu_count() // 2, args.num_envs)  # Use half of CPU cores, max 4
            print(f"Processing {len(tasks)} tasks with {num_workers} workers...")

            # Create a partial function with fixed config_path, model and debug
            process_func = partial(process_task, 
                                path_to_vm=args.path_to_vm,
                                snapshot_name=args.snapshot_name,
                                result_dir=args.result_dir,
                                coding_model=args.coding_model,
                                orchestrator_model=args.orchestrator_model,
                                config_path=args.oai_config_path, 
                                orchestrator_max_steps=args.orchestrator_max_steps,
                                cua_max_steps=args.cua_max_steps,
                                coding_max_steps=args.coding_max_steps,
                                cut_off_steps=args.cut_off_steps,
                                screen_width=args.screen_width,
                                screen_height=args.screen_height,
                                sleep_after_execution=args.sleep_after_execution,
                                client_password=args.client_password,
                                rag=args.rag,
                                rag_topk=args.rag_topk,
                                rag_filename=args.rag_filename
                                )

            # Process tasks in parallel
            with Pool(processes=num_workers) as pool:
                results = pool.map(process_func, tasks)

            # Collect scores from results
            for domain, score in results:
                scores[domain].append(score)

            # Print summary
            print("\n=== Task Processing Complete ===")
            for domain in scores:
                if scores[domain]:
                    avg_score = sum(scores[domain]) / len(scores[domain])
                    print(f"{domain}: {len(scores[domain])} tasks, average score: {avg_score:.2%}")
    
    all_scores = []
    count_remain = 0
    for domain in test_all_meta:
        for ex_id in test_all_meta[domain]:
            score_file = os.path.join(args.result_dir, f"{domain}/{ex_id}/result.txt")
            if os.path.exists(score_file):
                with open(score_file, "r") as f:
                    all_scores.append(float(f.read()))
            else:
                all_scores.append(0.0)
                count_remain += 1
    all_avg_score = np.mean(all_scores)
    print(f"All average score: {all_avg_score:.2%}, tasks remain: {count_remain}")
