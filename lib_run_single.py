#coding=utf8
import datetime, json, logging, os, argparse
from desktop_env.envs.desktop_env import DesktopEnv
from mm_agents.agent import PromptAgent
from wrapt_timeout_decorator import *
from utils import serialize_json

logger = logging.getLogger("desktopenv.experiment")

TIME_LIMIT = 1800 # 30 minutes for each example at most

@timeout(TIME_LIMIT, use_signals=False)
def run_single_example(agent: PromptAgent, env: DesktopEnv, example: dict, result_dir: str, args: argparse.Namespace) -> float:
    done, step_idx = False, 0
    agent.reset()
    obs = env.reset(task_config=example)
    infos = []
    # env.controller.start_recording()
    screenshots = os.path.join(result_dir, "screenshots")
    a11y_tree = os.path.join(result_dir, "a11y_trees")
    
    # Initialize action logs for statistics tracking
    action_logs = []
    chat_history = []

    while not done and step_idx < args.max_steps:
        context = example['context'] if 'context' in example else None
        response, actions = agent.predict(example['instruction'], obs, context)
        
        # Record the agent's response for chat history
        chat_history.append({
            "role": "assistant",
            "content": response,
            "step": step_idx + 1
        })
        
        infos = []
        for action_idx, action in enumerate(actions):
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
            logger.info("[Action]: Step %d: %s", step_idx + 1, action)

            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            infos.append(info) # add action execution result to the observation
            
            # Record action log for statistics
            action_log = {
                "step": step_idx + 1,
                "action_index": action_idx,
                "type": "gui_operator",  # spider2v agent primarily does GUI operations
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "cost": 0.0,  # Individual action cost (will be calculated later)
                "screenshot": f"step_{step_idx + 1}_{action_timestamp}.png" if args.observation_space != 'a11y_tree' else None
            }
            action_logs.append(action_log)

            # Save screenshot and trajectory information
            if args.observation_space != 'a11y_tree':
                with open(os.path.join(screenshots, f"step_{step_idx + 1}_{action_timestamp}.png"), "wb") as _f:
                    _f.write(agent.observations[-1]["raw_screenshot"])
            if args.observation_space != 'screenshot':
                with open(os.path.join(a11y_tree, f"step_{step_idx + 1}_{action_timestamp}.txt"), "w", encoding='utf-8') as _f:
                    _f.write(agent.observations[-1]["accessibility_tree"])
            # # write trajectory information (can replay the trajectory later)
            # with open(os.path.join(result_dir, "trajectory.jsonl"), "a") as f:
            #     f.write(json.dumps({
            #         "step_num": step_idx + 1,
            #         "action_timestamp": action_timestamp,
            #         "action": action,
            #         "reward": reward,
            #         "done": done,
            #         "info": info
            #     }) + '\n')
            if done:
                logger.info("[INFO]: The episode is done. Congratulations!")
                break
        if done: break
        obs['infos'] = infos
        step_idx += 1
    else:
        logger.warning("[WARNING]: Exceeded the maximum number of steps. Forced to stop the episode.")

    agent.get_current_cost()
    
    # Calculate total cost from agent usage
    
    total_cost = pc * agent.usages["prompt_tokens"] + cc * agent.usages["completion_tokens"]
    
    try: # for safety reason, wrap the evaluation in a try-except block
        result = env.evaluate()
    except Exception as e:
        error_msg = f"[ERROR]: Unexpected error occurred when evaluating the result: {e}"
        logger.error(error_msg)
        result = 0.0

    additional_context = example.get('context', '')
    
    unified_log = {
        "statistics": {
            "score": result,
            "total_steps": step_idx,
            "total_cost": total_cost,
            "prompt_tokens": agent.usages["prompt_tokens"],
            "completion_tokens": agent.usages["completion_tokens"],
        },
        "task_config": example,
        "additional_context": additional_context,
        "chat_history": chat_history,
        "action_logs": action_logs,
    }
    
    # Save unified execution log
    with open(os.path.join(result_dir, "execution_log.json"), "w", encoding="utf-8") as f:
        json.dump(serialize_json(unified_log), f, indent=2)

    logger.info(f"[Result]: Evaluation score for {example['id']}: {result:.1f}")
    logger.info(f"[Statistics]: Total steps: {step_idx+1}, GUI operations: {len(action_logs)}, Total cost: ${total_cost:.4f}")
    
    with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"{result}\n")

    # env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
    return result