"""Script to run end-to-end evaluation on the benchmark.
Utils and basic architecture credit to https://github.com/web-arena-x/webarena/blob/main/run.py.
"""
import argparse, datetime, json, logging, os, shutil, sys
from tqdm import tqdm
import lib_run_single
from typing import List, Tuple, Dict, Any, Optional 
from desktop_env.envs.desktop_env import DesktopEnv
from mm_agents.agent import PromptAgent
from utils import build_additional_contexts, summary, save_args_to_settings

#  Logger Configs {{{ #
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
os.makedirs("logs", exist_ok=True)
datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
file_handler = logging.FileHandler(os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8")
debug_handler = logging.FileHandler(os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)
file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
pure_formatter = logging.Formatter(fmt="[%(asctime)s %(levelname)s %(module)s/%(lineno)d]: %(message)s")
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
stdout_handler.addFilter(logging.Filter("desktopenv"))
logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger = logging.getLogger("desktopenv.experiment")


ALL_DOMAINS = ['excel', 'servicenow', 'jupyter', 'dbt', 'airflow', 'dagster', 'airbyte', 'snowflake', 'bigquery', 'superset', 'metabase']

def get_retrieved_context(config_path: str, topk: int = 4, file_name: str = "retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt") -> str:
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

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation on the benchmark")

    # environment config
    parser.add_argument('-p', "--path_to_vm", type=str, help="path to the VM executable .vmx file, if None, automatically find the VM in vm_data/ folder")
    parser.add_argument('-s', "--snapshot_name", type=str, default="init_state", help="Snapshot name to use (overwrite snapshot in each example config)")
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument(
        "--action_space",
        choices=[
            "pyautogui",
            "computer_13"
        ],
        default="pyautogui",
        help="Action space to use for the agent"
    )
    parser.add_argument(
        "--observation_space",
        choices=[
            "screenshot",
            "a11y_tree",
            "screenshot_a11y_tree",
            "som"
        ],
        default="som",
        help="Observation space to use for the environment",
    )
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument("--max_steps", type=int, default=15, help="Maximum number of steps for each example, this can be altered dynamically according to field `action_number` in the example config")

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3, help='maximum length of interaction history to provide to the agent')
    parser.add_argument("--a11y_tree_max_tokens", type=int, default=5000, help='maximum length of interaction history to provide to the agent')

    # llm config
    parser.add_argument('-m', "--model", type=str, default="gpt-4o-2024-05-13", help="LLM model to use for the agent")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)

    # example config
    parser.add_argument('-e', "--example", type=str, default=os.path.join('evaluation_examples', 'test_one.json'), help="JSON dict containing example ids to run")
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples/examples"
    )
    parser.add_argument("--exclude_account", action='store_true')
    parser.add_argument("--execution_feedback", action='store_true', help="whether to use execution feedback for the agent")
    parser.add_argument("--rag", action='store_true', help="Whether to use RAG for the agent")
    parser.add_argument("--rag_topk", type=int, default=4, help="Top k to use for RAG")
    parser.add_argument("--rag_filename", type=str, default="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt", help="RAG retrieved context file name")
    parser.add_argument("--verbose_instruction", action='store_true', help="Enable verbose instruction loading")
    parser.add_argument("--domains", choices=ALL_DOMAINS + ['all'], nargs='+', default=["all"], help="Application names list to filter examples")

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results/som_gpt_4o_rag_ef")
    parser.add_argument("--rerun", action="store_true", help="Rerun tests that have already been run")
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores")
    args = parser.parse_args()

    if args.observation_space == 'som':
        assert args.action_space == 'pyautogui', "SOM only supports pyautogui action space"
    return args



def test(args: argparse.Namespace, test_all_meta: List[dict]) -> dict:
    scores = {}
    logger.info("Args: %s", args)
    env = DesktopEnv(
        path_to_vm=args.path_to_vm,
        snapshot_name=args.snapshot_name,
        action_space=args.action_space,
        headless=args.headless,
        require_a11y_tree=args.observation_space in ["a11y_tree", "screenshot_a11y_tree", "som"]
    )

    agent = PromptAgent(
        platform="ubuntu",
        model=args.model,
        max_tokens=args.max_tokens,
        action_space=args.action_space,
        observation_space=args.observation_space,
        execution_feedback=args.execution_feedback,
        screen_size=env.vm_screen_size,
        temperature=args.temperature,
        max_trajectory_length=args.max_trajectory_length,
        a11y_tree_max_tokens=args.a11y_tree_max_tokens
    )

    for example in tqdm(test_all_meta, desc="Example", leave=False):
        domain, eid = example['domain'], example['id']
        if domain not in scores:
            scores[domain] = []
        config_file, result_dir = example['config'], example['result']
        with open(config_file, "r", encoding="utf-8") as f:
            example = json.load(f)

        # Build context using the common function
        if args.rag: example['context'] = get_retrieved_context(config_file, args.rag_topk, file_name=args.rag_filename)
        else: example['context'] = None

        # root_logger = logging.getLogger()
        # example_handler = logging.FileHandler(os.path.join(result_dir, "result-{:}.log".format(datetime_str)), encoding="utf-8")
        # example_handler.setLevel(logging.INFO)
        # example_handler.setFormatter(pure_formatter)
        # example_handler.addFilter(logging.Filter("desktopenv"))
        # root_logger.addHandler(example_handler)

        logger.info(f"[Domain]: {domain}")
        logger.info(f"[Example id]: {eid}")
        logger.info(f"[Result dir]: {result_dir}")
        logger.info(f"[Instruction]: {example['instruction']}")

        # example start running
        try:
            score = lib_run_single.run_single_example(agent, env, example, result_dir, args)
            scores[domain].append(score)
        except Exception as e: # do not record in this case
            logger.error(f"Exception in {domain}/{eid}: {e}")
            # env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
            # with open(os.path.join(result_dir, "trajectory.jsonl"), "a") as f:
            #     f.write(json.dumps({
            #         "Error": f"Error msg in {domain}/{eid}: {e}"
            #     }))
            #     f.write("\n")

        # root_logger.removeHandler(example_handler)
        # example_handler.close()

    env.close()
    return scores

def get_examples(args, test_all_meta, easy_first: bool = True) -> List[Dict[str, str]]:
    """ Get [Filter] the list of example dict for the current experiment.
    # Filter method:
    - args.rerun (bool): if True, rerun tests that have already been run
    - args.rerun_fail (bool): if True, rerun failed tests
    - args.domains (List[str]): if not contain "all", only include examples under the specified domains
    - args.exclude_account (bool): if True, exclude examples that are related to real accounts
    - easy_first (bool): if True, sort examples that are easy to run first (smaller action_number)

    # The returned dict for each example in the List containing:
        - id: example id
        - domain: example domain, a.k.a., professional tool name
        - config: .json config path for the example
        - result: path to the result directory for the example
            note that, the result directory will also be reset implicitly
    """

    examples_to_run = []
    for domain in test_all_meta:
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
                os.makedirs(target_dir, exist_ok=True)
                if args.observation_space != "a11y_tree":
                    os.makedirs(os.path.join(target_dir, "screenshots"), exist_ok=True)
                if args.observation_space != "screenshot":
                    os.makedirs(os.path.join(target_dir, "a11y_trees"), exist_ok=True)
                example = {
                    "id": ex_id,
                    "domain": domain,
                    "config": cfg,
                    "result": target_dir,
                    "action_number":  json.load(open(cfg, 'r'))["action_number"]
                }
                examples_to_run.append(example)

    logger.info(f"Total examples to run: {len(examples_to_run)}")
    if easy_first:
        sorted(examples_to_run, key=lambda x: x['action_number'])
    return examples_to_run


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()

    save_args_to_settings(args, args.result_dir)
    test_all_meta = json.load(open(args.example, 'r'))
    examples = get_examples(args, test_all_meta)

    if not args.get_score:
        test(args, examples)
    
    summary(args, test_all_meta)