#coding=utf8
import datetime, json, logging, os, argparse
from desktop_env.envs.desktop_env import DesktopEnv
from mm_agents.agent import PromptAgent
from wrapt_timeout_decorator import *
from utils import serialize_json

logger = logging.getLogger("desktopenv.experiment")


def filter_base64_images_from_messages(messages: list) -> list:
    """Remove base64 images from messages for logging purposes."""
    filtered_messages = []
    for msg in messages:
        filtered_msg = {"role": msg["role"]}
        content = msg["content"]
        
        if isinstance(content, str):
            # String content - keep as is
            filtered_msg["content"] = content
        elif isinstance(content, list):
            # List content - filter out images
            filtered_content = []
            for item in content:
                if item.get("type") == "input_text":
                    filtered_content.append(item)
                elif item.get("type") == "input_image":
                    # Replace image with placeholder
                    filtered_content.append({
                        "type": "input_image",
                    })
            filtered_msg["content"] = filtered_content
        else:
            filtered_msg["content"] = content
        
        filtered_messages.append(filtered_msg)
    
    return filtered_messages


def merge_conversation_and_actions(conversation_history: list, action_logs: list) -> list:
    """Merge conversation messages and action logs into a unified timeline."""
    timeline = []
    
    # Filter conversation messages to remove base64 images
    filtered_messages = filter_base64_images_from_messages(conversation_history)
    
    # Group messages into pairs (user + assistant)
    message_pairs = []
    i = 0
    while i < len(filtered_messages):
        if filtered_messages[i]["role"] == "user":
            user_msg = filtered_messages[i]
            assistant_msg = filtered_messages[i + 1] if i + 1 < len(filtered_messages) else None
            message_pairs.append({"user": user_msg, "assistant": assistant_msg})
            i += 2
        else:
            i += 1
    
    # Match conversation pairs with action logs
    # First pair is initial task instruction (step 0)
    if message_pairs:
        first_pair = message_pairs[0]
        timeline.append({
            "step": 0,
            "type": "task_instruction",
            "user_message": first_pair["user"]["content"],
            "assistant_response": first_pair["assistant"]["content"] if first_pair["assistant"] else None
        })
    
    # Process remaining pairs with corresponding actions
    for idx, pair in enumerate(message_pairs[1:], start=1):
        # Add action log if available
        if idx - 1 < len(action_logs):
            action = action_logs[idx - 1].copy()
            # Add conversation context to action
            action["coordinator_decision"] = pair["assistant"]["content"] if pair["assistant"] else None
            action["execution_result"] = pair["user"]["content"] if idx < len(message_pairs) else None
            timeline.append(action)
        else:
            # No action log, just record the conversation
            timeline.append({
                "step": idx,
                "type": "conversation_only",
                "user_message": pair["user"]["content"],
                "assistant_response": pair["assistant"]["content"] if pair["assistant"] else None
            })
    
    # Add any remaining action logs that don't have conversation pairs
    for idx in range(len(message_pairs) - 1, len(action_logs)):
        timeline.append(action_logs[idx])
    
    return timeline

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
    
    # Add initial task instruction to chat history
    chat_history.append({
        "role": "user",
        "content": example['instruction']
    })

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
    
    # Merge conversation history and action logs into unified timeline
    unified_timeline = merge_conversation_and_actions(chat_history, action_logs)
    
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
        "action_logs": unified_timeline,
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