#!/usr/bin/env python3
"""
LangGraph-based Dual Agent Framework
Complete implementation with proper state management and debugging
"""

import base64
import json
import os
import logging
import time
import traceback
from typing import TypedDict, Annotated, Literal, Optional, Tuple, List, Any, Dict

import operator as op
from langgraph.graph import StateGraph, END
from openai import OpenAI
import openai

from desktop_env.envs.desktop_env import DesktopEnv
from configs.config import OPENAI_API_KEY
from utils import serialize_json, get_price


PYTHON_VARIANTS = ["python", "Python", "py"]


# ==================== PROMPTS ====================

COORDINATOR_SYSTEM_MESSAGE = """# Your role
You are a task solver, you need to complete a computer-using task step-by-step.
1. Describe the screenshot.
2. Provide a detailed plan, including a list of user requirements like specific file name, file path, etc.
3. Follow the following instructions and complete the task with your skills.
    - If you think the task is impossible to complete (no file, wrong environment, etc.), reply with "INFEASIBLE" to end the conversation.
    - **Do not** do anything else out of the user's instruction like change the file name. This will make the task fail.
    - Check every screenshot carefully and see if it fulfills the task requirement.
    - You MUST try code execution first for file operation tasks like spreadsheet modification.
4. Verify the result and see if it fulfills the user's requirement.

# Your skills
You can use the following tools to solve the task:

## Code Execution
You can write code in ```bash...``` code blocks for bash scripts, and ```python...``` code blocks for python code.
- Your linux username is "user"
- If you want to use sudo, follow the format: "echo {CLIENT_PASSWORD} | sudo -S [YOUR COMMANDS]" (no quotes for the word "{CLIENT_PASSWORD}")
- You MUST verify the result before save the changes
- When you write code, you must identify the language (whether it is python or bash) of the code
- Wrap all your code in ONE code block. DO NOT let user save the code as a file and execute it for you
- Do not include __main__ in your python code
- When you modify a spreadsheet, **make sure every value is in the expected cell**
- When importing a package, you need to check if the package has been installed. If not, you need to install it yourself
- You need to print the progressive and final result
- If you met execution error, you need to analyze the error message and try to fix the error

## GUI Operator
Let a GUI Operator to solve a subtask you assigned.
GUI Operator can operate the computer by clicking and typing (but not accurate).
Require a detailed task description.
The Operator executes ONE GUI action per call (one click, one type, etc.). Each step is a one-time interaction with OS like mouse click or keyboard typing. Please take this into account when you plan the actions.
If you let GUI Operator to check the result, you MUST let it close and reopen the file because programmer's result will NOT be updated to the screen.

Remember: Use code execution for file operations and data processing. Use GUI Operator for visual interactions that require precise clicking or typing.

# Decision Format
Your response MUST be in JSON format:
{
    "thought": "Your reasoning about current situation and next action",
    "action": "code|gui|terminate",
    "content": "Code to execute OR task description for GUI operator",
    "language": "python|bash (only for code action)"
}

If task is completed successfully, use action "terminate" with thought explaining completion.
If task is impossible, use action "terminate" with thought "INFEASIBLE: [reason]".
"""

OPERATOR_SYSTEM_MESSAGE = """# Task
{instruction}

# Hints
- Sudo password is "{CLIENT_PASSWORD}".
- Execute only the specific GUI action requested.
- Use precise coordinates for clicking and typing.
"""


# ==================== HELPER FUNCTIONS ====================

def call_openai_cua(
    client: OpenAI,
    history_inputs: list,
    screen_width: int = 1920,
    screen_height: int = 1080,
    environment: str = "linux",
    model: str = "computer-use-preview",
    logger: logging.Logger = None
) -> Tuple[Any, float, int, int]:
    """Call OpenAI Computer Use API with retry logic."""
    retry = 0
    response = None
    
    while retry < 3:
        try:
            response = client.responses.create(
                model=model,
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": screen_width,
                    "display_height": screen_height,
                    "environment": environment,
                }],
                input=history_inputs,
                reasoning={"summary": "concise"},
                tool_choice="required",
                truncation="auto",
            )
            break
        except openai.BadRequestError as e:
            retry += 1
            if logger:
                logger.error(f"BadRequestError in response.create (retry {retry}): {e}")
            time.sleep(0.5)
        except openai.InternalServerError as e:
            retry += 1
            if logger:
                logger.error(f"InternalServerError in response.create (retry {retry}): {e}")
            time.sleep(0.5)
        except Exception as e:
            retry += 1
            if logger:
                logger.error(f"Error in response.create (retry {retry}): {e}")
            time.sleep(0.5)
    
    if retry == 3:
        raise Exception("Failed to call OpenAI after 3 retries.")

    cost = 0.0
    input_tokens = 0
    output_tokens = 0
    
    try:
        prompt_price, completion_price = get_price(model)
        if response and hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            input_cost = input_tokens * prompt_price
            output_cost = output_tokens * completion_price
            cost = input_cost + output_cost
    except Exception as e:
        if logger:
            logger.warning(f"Error calculating cost: {e}")

    return response, cost, input_tokens, output_tokens


def _cua_to_pyautogui(action) -> str:
    """Convert OpenAI CUA action to pyautogui command."""
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
    
    return "WAIT"


# ==================== STATE DEFINITION ====================

class AgentState(TypedDict):
    """LangGraph state for the dual agent workflow."""
    # Task information
    task_instruction: str
    additional_context: str
    task_config: dict
    
    # Current state
    current_screenshot: bytes
    operation_count: int  # Count of actual operations (code + gui)
    
    # Coordinator decision
    coordinator_thought: str
    coordinator_action: str
    coordinator_content: str
    coordinator_language: str
    
    # Execution results
    last_execution_result: str
    last_execution_success: bool
    
    # History and logs
    action_logs: Annotated[List[dict], op.add]
    conversation_history: Annotated[List[dict], op.add]
    
    # Tracking
    model_usage: dict
    
    # Control flow
    next_node: str
    task_completed: bool
    task_infeasible: bool
    
    # Environment reference
    env: Optional[DesktopEnv]
    operations_dir: str
    
    # Configuration
    coordinator_model: str
    operator_model: str
    client_password: str
    screen_width: int
    screen_height: int
    sleep_after_execution: float
    max_steps: int


# ==================== NODE FUNCTIONS ====================

def coordinator_node(state: AgentState) -> dict:
    """Coordinator makes decisions about next action."""
    logger = logging.getLogger("desktopenv")
    
    screenshot = state["current_screenshot"]
    screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
    
    # Prepare prompt
    if state["operation_count"] == 0:
        user_message = f"""{state['task_instruction']}{state['additional_context']}

Check my computer screenshot and describe it first. If this task is possible to complete, please complete it on my computer. If not, reply with "INFEASIBLE" to end the conversation.

Current screenshot attached below."""
    else:
        user_message = f"""Previous action result:
{state['last_execution_result']}

Continue with the task or verify if completed. Current screenshot attached below."""
    
    # Build messages with history
    messages = [{"role": "system", "content": COORDINATOR_SYSTEM_MESSAGE}]
    
    # Add recent conversation history (last 6 messages = 3 turns)
    history = state.get("conversation_history", [])
    if len(history) > 6:
        messages.extend(history[-6:])
    else:
        messages.extend(history)
    
    # Add current message
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_message},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
        ]
    })
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[Coordinator] Operation count: {state['operation_count']}")
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=state.get("coordinator_model"),
            messages=messages,
            max_completion_tokens=2000,
        )
        
        response_text = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        # Calculate cost
        prompt_price, completion_price = get_price(state.get("coordinator_model"))
        cost = (prompt_tokens * prompt_price + completion_tokens * completion_price)
        
        # Update model usage
        model_usage = state.get("model_usage", {}).copy()
        if "coordinator" not in model_usage:
            model_usage["coordinator"] = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
        model_usage["coordinator"]["cost"] += cost
        model_usage["coordinator"]["prompt_tokens"] += prompt_tokens
        model_usage["coordinator"]["completion_tokens"] += completion_tokens
        
        # Parse JSON response
        try:
            decision = json.loads(response_text)
        except json.JSONDecodeError:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                decision = json.loads(response_text[json_start:json_end].strip())
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                decision = json.loads(response_text[json_start:json_end].strip())
            else:
                decision = {
                    "thought": response_text,
                    "action": "terminate",
                    "content": response_text,
                    "language": "python"
                }
        
        logger.info(f"Thought: {decision.get('thought', 'N/A')[:200]}")
        logger.info(f"Action: {decision.get('action', 'N/A')}")
        logger.info(f"Content preview: {str(decision.get('content', ''))[:100]}")
        
        # Add to conversation history
        conversation_history = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response_text}
        ]
        
        # Determine next node
        action = decision.get("action", "terminate").lower()
        
        if action == "terminate":
            task_infeasible = "infeasible" in decision.get("thought", "").lower()
            logger.info(f"Task {'INFEASIBLE' if task_infeasible else 'COMPLETED'}")
            return {
                "coordinator_thought": decision.get("thought", ""),
                "coordinator_action": action,
                "coordinator_content": decision.get("content", ""),
                "coordinator_language": decision.get("language", "python"),
                "conversation_history": conversation_history,
                "model_usage": model_usage,
                "next_node": "evaluator",
                "task_completed": True,
                "task_infeasible": task_infeasible
            }
        elif action == "code":
            logger.info("Next: Code Execution")
            return {
                "coordinator_thought": decision.get("thought", ""),
                "coordinator_action": action,
                "coordinator_content": decision.get("content", ""),
                "coordinator_language": decision.get("language", "python"),
                "conversation_history": conversation_history,
                "model_usage": model_usage,
                "next_node": "code_executor",
                "task_completed": False,
                "task_infeasible": False
            }
        elif action == "gui":
            logger.info("Next: GUI Operation")
            return {
                "coordinator_thought": decision.get("thought", ""),
                "coordinator_action": action,
                "coordinator_content": decision.get("content", ""),
                "coordinator_language": decision.get("language", "python"),
                "conversation_history": conversation_history,
                "model_usage": model_usage,
                "next_node": "gui_operator",
                "task_completed": False,
                "task_infeasible": False
            }
        else:
            logger.warning(f"Unknown action: {action}, terminating")
            return {
                "coordinator_thought": decision.get("thought", ""),
                "coordinator_action": "terminate",
                "coordinator_content": "",
                "conversation_history": conversation_history,
                "model_usage": model_usage,
                "next_node": "evaluator",
                "task_completed": True,
                "task_infeasible": False
            }
        
    except Exception as e:
        logger.error(f"Coordinator error: {e}")
        logger.error(traceback.format_exc())
        return {
            "coordinator_thought": f"Error: {str(e)}",
            "coordinator_action": "terminate",
            "coordinator_content": "",
            "next_node": "evaluator",
            "task_completed": True,
            "task_infeasible": True
        }


def code_executor_node(state: AgentState) -> dict:
    """Execute code (Python or Bash) in the desktop environment."""
    logger = logging.getLogger("desktopenv")
    
    code = state["coordinator_content"]
    lang = state["coordinator_language"]
    env = state["env"]
    operation_count = state["operation_count"] + 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[Code Execution] Operation #{operation_count}")
    logger.info(f"Language: {lang}")
    logger.info(f"Code:\n{code}")
    logger.info("="*80)
    
    exitcode = 1
    logs = ""
    
    try:
        if lang in ["bash", "shell", "sh"]:
            output_dict = env.controller.run_bash_script(code, timeout=300)
            exitcode = 0 if output_dict["status"] == "success" else 1
            logs = output_dict["output"]
        elif lang in PYTHON_VARIANTS:
            output_dict = env.controller.run_python_script(code, timeout=300)
            exitcode = 0 if output_dict["status"] != "error" else 1
            logs = output_dict.get("message", output_dict.get("output", ""))
        else:
            exitcode = 1
            logs = f"Unsupported language: {lang}"
    except Exception as e:
        exitcode = 1
        logs = f"Execution error: {str(e)}"
        logger.error(traceback.format_exc())
    
    success = (exitcode == 0)
    
    # Take screenshot after execution
    screenshot = env.controller.get_screenshot()
    screenshot_filename = f"step_{operation_count}_code.png"
    
    with open(os.path.join(state["operations_dir"], screenshot_filename), "wb") as f:
        f.write(screenshot)
    
    # Log the action
    action_log = {
        "step": operation_count,
        "type": "code_execution",
        "language": lang,
        "code": code,
        "exitcode": exitcode,
        "output": logs,
        "cost": 0.0,
        "screenshot": screenshot_filename
    }
    
    result_message = f"Code execution {'succeeded' if success else 'failed'}.\nOutput:\n{logs}"
    logger.info(f"Exit code: {exitcode}")
    logger.info(f"Output: {logs[:500]}")
    
    return {
        "operation_count": operation_count,
        "current_screenshot": screenshot,
        "last_execution_result": result_message,
        "last_execution_success": success,
        "action_logs": [action_log],
        "next_node": "coordinator"
    }


def gui_operator_node(state: AgentState) -> dict:
    """Execute GUI operation using OpenAI's computer use API."""
    logger = logging.getLogger("desktopenv")
    
    task = state["coordinator_content"]
    context = state["coordinator_thought"]
    env = state["env"]
    operation_count = state["operation_count"] + 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[GUI Operation] Operation #{operation_count}")
    logger.info(f"Task: {task}")
    logger.info("="*80)
    
    screenshot = state["current_screenshot"]
    screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
    
    # Save screenshot before operation
    screenshot_filename = f"step_{operation_count}_gui.png"
    with open(os.path.join(state["operations_dir"], screenshot_filename), "wb") as f:
        f.write(screenshot)
    
    # Prepare operator prompt
    task_instruction = f"""# GUI Action Request
{task}

# Context
{context}

# Instructions
Analyze the current screenshot and execute the requested GUI operation."""
    
    operator_prompt = OPERATOR_SYSTEM_MESSAGE.format(
        instruction=task_instruction,
        CLIENT_PASSWORD=state.get("client_password", "password")
    )
    
    history_inputs = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": operator_prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"},
        ],
    }]
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response, cost, input_tokens, output_tokens = call_openai_cua(
            client, 
            history_inputs, 
            state.get("screen_width", 1920), 
            state.get("screen_height", 1080),
            model=state.get("operator_model", "computer-use-preview"),
            logger=logger
        )
        
        reasoning = ""
        executed_action = False
        
        for output_item in response.output:
            output_type = output_item.get("type", "") if isinstance(output_item, dict) else getattr(output_item, "type", "")
            
            if "computer_call" in str(output_type):
                action_call = output_item if isinstance(output_item, dict) else output_item.model_dump()
                py_cmd = _cua_to_pyautogui(action_call["action"])
                
                obs, *_ = env.step(py_cmd, state.get("sleep_after_execution", 0.5))
                executed_action = True
                logger.info(f"Executed: {py_cmd}")
                
            elif "reasoning" in str(output_type) and hasattr(output_item, 'summary') and len(output_item.summary) > 0:
                reasoning = output_item.summary[0].text
                logger.info(f"Reasoning: {reasoning}")
                
            elif "message" in str(output_type):
                message_text = output_item.content[0].text if hasattr(output_item, 'content') else str(output_item)
                reasoning = message_text
        
        if not reasoning:
            reasoning = "Executed GUI action" if executed_action else "No action executed"
        
        # Update model usage
        model_usage = state.get("model_usage", {}).copy()
        if "operator" not in model_usage:
            model_usage["operator"] = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
        model_usage["operator"]["cost"] += cost
        model_usage["operator"]["prompt_tokens"] += input_tokens
        model_usage["operator"]["completion_tokens"] += output_tokens
        
        # Get updated screenshot
        screenshot = env.controller.get_screenshot()
        
        action_log = {
            "step": operation_count,
            "type": "gui_operator",
            "task": task,
            "context": context,
            "result": reasoning,
            "cost": cost,
            "model": state.get("operator_model", "computer-use-preview"),
            "screenshot": screenshot_filename
        }
        
        logger.info(f"Success: {executed_action}")
        
        return {
            "operation_count": operation_count,
            "current_screenshot": screenshot,
            "last_execution_result": f"GUI operation completed: {reasoning}",
            "last_execution_success": executed_action,
            "action_logs": [action_log],
            "model_usage": model_usage,
            "next_node": "coordinator"
        }
        
    except Exception as e:
        logger.error(f"GUI operation error: {e}")
        logger.error(traceback.format_exc())
        
        action_log = {
            "step": operation_count,
            "type": "gui_operator",
            "task": task,
            "context": context,
            "result": f"Error: {str(e)}",
            "cost": 0.0,
            "screenshot": screenshot_filename
        }
        
        return {
            "operation_count": operation_count,
            "last_execution_result": f"GUI operation failed: {str(e)}",
            "last_execution_success": False,
            "action_logs": [action_log],
            "next_node": "coordinator"
        }


def evaluator_node(state: AgentState) -> dict:
    """Evaluate task completion and calculate final score."""
    logger = logging.getLogger("desktopenv")
    env = state["env"]
    
    logger.info(f"\n{'='*80}")
    logger.info("Task Evaluation")
    logger.info("="*80)
    
    try:
        score = env.evaluate()
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        score = 0.0
    
    # Calculate statistics
    cua_steps = len([log for log in state["action_logs"] if log["type"] == "gui_operator"])
    code_steps = len([log for log in state["action_logs"] if log["type"] == "code_execution"])
    
    model_usage = state.get("model_usage", {})
    prompt_tokens = sum(model_usage[model]["prompt_tokens"] for model in model_usage)
    completion_tokens = sum(model_usage[model]["completion_tokens"] for model in model_usage)
    total_cost = sum(model_usage[model]["cost"] for model in model_usage)
    
    logger.info(f"Score: {score}")
    logger.info(f"Total operations: {state['operation_count']} (GUI: {cua_steps}, Code: {code_steps})")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion")
    
    # Create execution log
    execution_log = {
        "statistics": {
            "score": score,
            "total_steps": state["operation_count"],
            "cua_steps": cua_steps,
            "coding_steps": code_steps,
            "total_cost": total_cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model_usage": model_usage,
            "task_infeasible": state.get("task_infeasible", False)
        },
        "task_config": state["task_config"],
        "additional_context": state["additional_context"],
        "action_logs": state["action_logs"],
        "final_thought": state.get("coordinator_thought", "")
    }
    
    # Save execution log
    history_save_dir = os.path.dirname(state["operations_dir"])
    with open(os.path.join(history_save_dir, "execution_log.json"), "w") as f:
        json.dump(serialize_json(execution_log), f, indent=2)
    
    # Save score
    with open(os.path.join(history_save_dir, "result.txt"), "w") as f:
        f.write(str(score))
    
    logger.info("="*80 + "\n")
    
    return {
        "next_node": "done",
        "task_completed": True
    }


# ==================== ROUTING ====================

def route_next(state: AgentState) -> Literal["coordinator", "code_executor", "gui_operator", "evaluator", "done"]:
    """Route to next node based on current state."""
    
    # Check max steps
    if state["operation_count"] >= state.get("max_steps", 15):
        return "evaluator"
    
    # Check task completion
    if state.get("task_completed", False):
        return "done" if state["next_node"] == "done" else "evaluator"
    
    # Route based on next_node
    return state.get("next_node", "coordinator")


# ==================== WORKFLOW ====================

def create_workflow() -> StateGraph:
    """Create the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("code_executor", code_executor_node)
    workflow.add_node("gui_operator", gui_operator_node)
    workflow.add_node("evaluator", evaluator_node)
    
    workflow.set_entry_point("coordinator")
    
    workflow.add_conditional_edges(
        "coordinator",
        route_next,
        {
            "coordinator": "coordinator",
            "code_executor": "code_executor",
            "gui_operator": "gui_operator",
            "evaluator": "evaluator",
            "done": END
        }
    )
    
    workflow.add_conditional_edges(
        "code_executor",
        route_next,
        {"coordinator": "coordinator", "evaluator": "evaluator"}
    )
    
    workflow.add_conditional_edges(
        "gui_operator",
        route_next,
        {"coordinator": "coordinator", "evaluator": "evaluator"}
    )
    
    workflow.add_edge("evaluator", END)
    
    return workflow.compile()


# ==================== FRAMEWORK ====================

class MyAgentFramework:
    """LangGraph-based dual agent framework."""
    
    def __init__(
        self,
        coordinator_model: str = "gpt-4-turbo-preview",
        operator_model: str = "computer-use-preview",
        operator_client_password: str = "",
        screen_width: int = 1920,
        screen_height: int = 1080,
        sleep_after_execution: float = 0.5,
        llm_config_path: str = None,
        max_steps: int = 15,
        history_save_dir: str = ""
    ):
        self.coordinator_model = coordinator_model
        self.operator_model = operator_model
        self.client_password = operator_client_password
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sleep_after_execution = sleep_after_execution
        self.max_steps = max_steps
        self.history_save_dir = history_save_dir
        
        self.workflow = create_workflow()
        self.env = None
    
    def setup_environment(
        self, 
        path_to_vm: str,
        snapshot_name: str = "init_state",
        headless: bool = False
    ):
        """Initialize the desktop environment."""
        self.env = DesktopEnv(
            path_to_vm=path_to_vm,
            action_space="pyautogui",
            snapshot_name=snapshot_name,
            headless=headless,
            require_a11y_tree=False
        )
    
    def execute_task(
        self,
        task_config: dict,
        additional_context: Optional[str] = None
    ) -> float:
        """
        Execute a task using the LangGraph workflow.
        
        Args:
            task_config: Task configuration
            additional_context: Optional additional context
            
        Returns:
            Task score
        """
        if not self.env:
            raise ValueError("Environment not initialized. Call setup_environment() first.")
        
        logger = logging.getLogger("desktopenv")
        
        # Reset environment
        self.env.reset(task_config=task_config)
        
        # Create operations directory
        operations_dir = os.path.join(self.history_save_dir, "operations")
        os.makedirs(operations_dir, exist_ok=True)
        
        # Get initial screenshot
        screenshot = self.env.controller.get_screenshot()
        
        # Initialize state
        initial_state = {
            "task_instruction": task_config["instruction"],
            "additional_context": additional_context or "",
            "task_config": task_config,
            "current_screenshot": screenshot,
            "operation_count": 0,
            "coordinator_thought": "",
            "coordinator_action": "",
            "coordinator_content": "",
            "coordinator_language": "python",
            "last_execution_result": "",
            "last_execution_success": True,
            "action_logs": [],
            "conversation_history": [],
            "model_usage": {},
            "next_node": "coordinator",
            "task_completed": False,
            "task_infeasible": False,
            "env": self.env,
            "operations_dir": operations_dir,
            "coordinator_model": self.coordinator_model,
            "operator_model": self.operator_model,
            "client_password": self.client_password,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "sleep_after_execution": self.sleep_after_execution,
            "max_steps": self.max_steps
        }
        
        logger.info("\n" + "="*80)
        logger.info("Starting LangGraph Workflow Execution")
        logger.info("="*80)
        logger.info(f"Task: {task_config['instruction'][:100]}...")
        logger.info(f"Coordinator model: {self.coordinator_model}")
        logger.info(f"Operator model: {self.operator_model}")
        logger.info(f"Max steps: {self.max_steps}")
        logger.info("="*80 + "\n")
        
        # Execute workflow with detailed logging
        try:
            iteration = 0
            for state_update in self.workflow.stream(initial_state):
                iteration += 1
                node_name = list(state_update.keys())[0]
                node_state = state_update[node_name]
                
                logger.debug(f"\n{'#'*80}")
                logger.debug(f"Iteration {iteration}: Node '{node_name}' completed")
                logger.debug(f"  Operation count: {node_state.get('operation_count', 0)}")
                logger.debug(f"  Next node: {node_state.get('next_node', 'N/A')}")
                logger.debug(f"  Task completed: {node_state.get('task_completed', False)}")
                logger.debug(f"  Coordinator action: {node_state.get('coordinator_action', 'N/A')}")
                logger.debug(f"{'#'*80}\n")
                
                # Stop if we've reached the end
                if node_name == "evaluator" or node_state.get("task_completed"):
                    final_state = node_state
                    break
            else:
                # If loop completes without break, get final state
                final_state = node_state
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            logger.error(traceback.format_exc())
            
            # Create emergency execution log
            execution_log = {
                "statistics": {
                    "score": 0.0,
                    "total_steps": 0,
                    "cua_steps": 0,
                    "coding_steps": 0,
                    "total_cost": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "model_usage": {},
                    "task_infeasible": True,
                    "error": str(e)
                },
                "task_config": task_config,
                "additional_context": additional_context or "",
                "action_logs": [],
                "final_thought": f"Workflow error: {str(e)}"
            }
            
            with open(os.path.join(self.history_save_dir, "execution_log.json"), "w") as f:
                json.dump(serialize_json(execution_log), f, indent=2)
            
            with open(os.path.join(self.history_save_dir, "result.txt"), "w") as f:
                f.write("0.0")
            
            return 0.0
        
        # Extract score from execution log
        execution_log_path = os.path.join(self.history_save_dir, "execution_log.json")
        if os.path.exists(execution_log_path):
            with open(execution_log_path, "r") as f:
                execution_log = json.load(f)
                score = execution_log["statistics"]["score"]
        else:
            logger.warning("Execution log not found, returning score 0.0")
            score = 0.0
        
        return score
    
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            print("Closing environment...")
            self.env.close()
            self.env = None
            print("Environment closed")