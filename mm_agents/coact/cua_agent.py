import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

from desktop_env.envs.desktop_env import DesktopEnv
from llm import AbstractLLM

logger = logging.getLogger("desktopenv")

PROMPT_TEMPLATE = """# Task
{instruction}

# Hints
- Sudo password is "{CLIENT_PASSWORD}".
- Keep the windows/applications opened at the end of the task.
- Do not use shortcut to reload the application except for the browser, just close and reopen.
- If "The document has been changed by others" pops out, you should click "cancel" and reopen the file.
- If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'.
- If you don't know how to continue the task, reply your concern or question along with 'IDK'.
""".strip()
DEFAULT_REPLY = "Please continue the user task. If you have completed the user task, reply with the information you want the user to know along with 'TERMINATE'."


def _cua_to_pyautogui(action) -> str:
    """Convert an Action (dict **or** Pydantic model) into a pyautogui call."""
    def fld(key: str, default: Any = None) -> Any:
        return action.get(key, default) if isinstance(action, dict) else getattr(action, key, default)

    act_type = fld("type")
    if not isinstance(act_type, str):
        act_type = str(act_type).split(".")[-1]
    act_type = act_type.lower()

    if act_type in ["click", "double_click"]:
        button = fld('button', 'left')
        if button == 1 or button == 'left':
            button = 'left'
        elif button == 2 or button == 'middle':
            button = 'middle'
        elif button == 3 or button == 'right':
            button = 'right'

        if act_type == "click":
            return f"pyautogui.click({fld('x')}, {fld('y')}, button='{button}')"
        if act_type == "double_click":
            return f"pyautogui.doubleClick({fld('x')}, {fld('y')}, button='{button}')"
        
    if act_type == "scroll":
        cmd = ""
        if fld('scroll_y', 0) != 0:
            cmd += f"pyautogui.scroll({-fld('scroll_y', 0) / 100}, x={fld('x', 0)}, y={fld('y', 0)});"
        return cmd
    if act_type == "drag":
        path = fld('path', [{"x": 0, "y": 0}, {"x": 0, "y": 0}])
        cmd = f"pyautogui.moveTo({path[0]['x']}, {path[0]['y']}, _pause=False); "
        cmd += f"pyautogui.dragTo({path[1]['x']}, {path[1]['y']}, duration=0.5, button='left')"
        return cmd

    if act_type == 'move':
        return f"pyautogui.moveTo({fld('x')}, {fld('y')})"

    if act_type == "keypress":
        keys = fld("keys", []) or [fld("key")]
        if len(keys) == 1:
            return f"pyautogui.press('{keys[0].lower()}')"
        else:
            return "pyautogui.hotkey('{}')".format("', '".join(keys)).lower()
        
    if act_type == "type":
        text = str(fld("text", ""))
        return "pyautogui.typewrite({:})".format(repr(text))
    
    if act_type == "wait":
        return "WAIT"
    
    return "WAIT"  # fallback


def _to_input_items(output_items: list) -> list:
    """
    Convert `response.output` into the JSON-serialisable items we're allowed
    to resend in the next request.  We drop anything the CUA schema doesn't
    recognise (e.g. `status`, `id`, …) and cap history length.
    """
    cleaned: List[Dict[str, Any]] = []

    for item in output_items:
        raw: Dict[str, Any] = item if isinstance(item, dict) else item.model_dump()

        # ---- strip noisy / disallowed keys ---------------------------------
        raw.pop("status", None)
        cleaned.append(raw)

    return cleaned  # keep just the most recent 50 items


def run_cua(
    env: DesktopEnv,
    instruction: str,
    max_steps: int,
    save_path: str = './',
    screen_width: int = 1920,
    screen_height: int = 1080,
    sleep_after_execution: float = 0.3,
    truncate_history_inputs: int = 100,
    client_password: str = "",
    model: str = "computer-use-preview",
) -> Tuple[List, str, float, int, int, int]:
    # Use AbstractLLM instead of direct OpenAI client
    llm = AbstractLLM(model, temperature=0, max_tokens=4096, logger=logger)

    # 0 / reset & first screenshot
    logger.info(f"Instruction: {instruction}")
    obs = env.controller.get_screenshot()
    screenshot_b64 = base64.b64encode(obs).decode("utf-8")
    with open(os.path.join(save_path, "initial_screenshot.png"), "wb") as f:
        f.write(obs)
    history_inputs = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": PROMPT_TEMPLATE.format(instruction=instruction, CLIENT_PASSWORD=client_password)},
            {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"},
        ],
    }]

    step_no = 0
    reasoning_list = []
    reasoning = ""
    llm.reset_stats()  # Reset stats at the beginning

    # 1 / iterative dialogue
    while step_no < max_steps:
        step_no += 1
        
        # Call CUA through AbstractLLM
        py_cmd, step_reasoning = llm.call_cua(
            messages=history_inputs,
            screen_width=screen_width,
            screen_height=screen_height,
            environment="linux"
        )
        
        if step_reasoning:
            reasoning_list.append(step_reasoning)
            logger.info(f"[Reasoning]: {step_reasoning}")
        
        # Check termination conditions
        if "TERMINATE" in py_cmd or "TERMINATE" in step_reasoning:
            reasoning = "My thinking process\n" + "\n- ".join(reasoning_list) + '\nPlease check the screenshot and see if it fulfills your requirements.'
            break
        
        if "IDK" in py_cmd or "IDK" in step_reasoning:
            reasoning = f"I don't know how to complete the task. Please check the current screenshot."
            break
        
        if py_cmd == "WAIT":
            logger.info("Waiting for next step")
            continue
        
        # Execute action
        logger.info(f"Executing: {py_cmd}")
        obs, *_ = env.step(py_cmd, sleep_after_execution)

        # Save screenshot
        screenshot_b64 = base64.b64encode(obs["screenshot"]).decode("utf-8")
        with open(os.path.join(save_path, f"step_{step_no}.png"), "wb") as f:
            f.write(obs["screenshot"])
        
        # Add screenshot to history for next iteration
        history_inputs.append({
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": f"Executed: {py_cmd}"}
            ]
        })
        history_inputs.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": DEFAULT_REPLY},
                {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"}
            ]
        })
        
        # Truncate history if needed (keep first message + recent messages)
        if len(history_inputs) > truncate_history_inputs:
            history_inputs = [history_inputs[0]] + history_inputs[-truncate_history_inputs:]
    
    logger.info("Task completed, press Esc to close the temporary window")
    esc_cmd = "pyautogui.press('esc')"
    obs, *_ = env.step(esc_cmd, sleep_after_execution)

    # Get cost and usage statistics from AbstractLLM
    total_cost, total_input_tokens, total_output_tokens, total_image_count = llm.get_usage()
    
    logger.info(f"Total cost for the task: ${total_cost:.4f}")
    logger.info(f"Total tokens: {total_input_tokens + total_output_tokens} (input: {total_input_tokens}, output: {total_output_tokens})")
    logger.info(f"Total images sent: {total_image_count}")
    
    # Clean image URLs in history for serialization
    for item in history_inputs:
        if isinstance(item.get('content'), list):
            for content_item in item['content']:
                if content_item.get('type') == 'input_image':
                    content_item['image_url'] = "<image>"
    
    return history_inputs, reasoning, total_cost, total_input_tokens, total_output_tokens, total_image_count