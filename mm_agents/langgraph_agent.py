#!/usr/bin/env python3
"""
LangGraph-based Dual Agent Framework with Direct GUI Feedback
Simplified version: directly tell operator what failed
"""

import base64
import json
import os
import logging
import traceback
from typing import TypedDict, Annotated, Literal, Optional, Tuple, List, Any, Dict
import operator as op
from langgraph.graph import StateGraph, END
from desktop_env.envs.desktop_env import DesktopEnv
from llm import AbstractLLM
from utils import serialize_json
from json_repair import repair_json


# ==================== PROMPTS ====================

COORDINATOR_SYSTEM_MESSAGE = """# Your Role
You are a task solver completing computer tasks step-by-step.

# Instructions
1. If task is impossible, reply with "INFEASIBLE" to end conversation
2. Check screenshots carefully to verify task completion
3. Prefer code execution for file operations and data processing
4. Do not modify user requirements (file names, paths, etc.)

# Available Tools
## Code Execution (Bash)
Execute bash commands in ```bash...``` blocks.
- Use sudo: "echo {CLIENT_PASSWORD} | sudo -S [COMMAND]"
- For Python: `python3 -c "code"` or install packages: `pip install numpy && python3 -c "import numpy"`
- Verify results before saving changes

## GUI Operator
Delegate GUI tasks to an operator (one action per call: one click, one type, etc.).
- Operator can click and type but positioning may be inaccurate
- If file modified by code, GUI must close and reopen file to see changes
- If an operator action produces wrong results, issue Ctrl+Z command to revert before trying alternative approach

## Handling GUI Failures
When GUI operations fail (check screenshot for expected result):
- **Consider using Ctrl+Z to undo the failed operation if it made unwanted changes**
- Provide HIGH-LEVEL guidance in "gui_feedback" (NO coordinates/positions)
- Suggest alternatives: keyboard shortcuts (including Ctrl+Z for undo), different elements, scrolling, etc.
- Describe visual landmarks: "button with save icon", "menu bar at top"
- After 2-3 failures of same subtask, consider switching to code approach

**Good feedback examples:**
- "Operation had wrong effect. Use Ctrl+Z to undo, then try clicking the correct menu item"
- "Menu didn't open. Try Alt+F keyboard shortcut instead"
- "Text not typed. Click the text field first to focus it"

**Bad feedback (never give coordinates):**
- "Click at (450, 320)" ✗
- "Move 50 pixels left" ✗

# Response Format (JSON)
{
    "thought": "Your reasoning about current situation and next action",
    "action": "code|gui|terminate",
    "content": "Bash commands OR GUI task description (can include Ctrl+Z command)",
    "gui_feedback": "Optional: high-level guidance if previous GUI failed (no coordinates)"
}

Use "terminate" action with "INFEASIBLE: [reason]" if task impossible."""

OPERATOR_SYSTEM_MESSAGE = """You are an agent that performs desktop computer tasks as instructed.
You will receive a screenshot and predict the action based on the image.

Use `pyautogui` to perform actions. DO NOT use `pyautogui.locateCenterOnScreen` or `pyautogui.screenshot()`.
Return Python code to perform ONE action at a time.
You must specify coordinates yourself based on the screenshot - be careful to ensure accuracy.

Return code in a code block:
```python
# your code here
```

My computer's password is '{CLIENT_PASSWORD}'.

{previous_attempt_info}

Analyze the screenshot carefully and return the code."""


# ==================== STATE DEFINITION ====================

class AgentState(TypedDict):
    """LangGraph state for the dual agent workflow."""
    # Task information
    task_instruction: str
    additional_context: str
    task_config: dict
    
    # Current state
    current_screenshot: bytes
    operation_count: int
    
    # Coordinator decision
    coordinator_thought: str
    coordinator_action: str
    coordinator_content: str
    
    # GUI tracking
    last_gui_task: str
    last_gui_command: str
    last_gui_success: Optional[bool]
    last_gui_result_check: str
    
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

    client = AbstractLLM(state.get("coordinator_model"), logger=logger)
    
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
    if client.is_vlm:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
            ]
        })
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
            ]
        })
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[Coordinator] Operation count: {state['operation_count']}")
    
    try:
        response_text = client(messages)
        cost, prompt_tokens, completion_tokens, image_count = client.get_usage()
        
        # Update model usage
        model_usage = state.get("model_usage", {}).copy()
        if "coordinator" not in model_usage:
            model_usage["coordinator"] = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
        model_usage["coordinator"]["cost"] += cost
        model_usage["coordinator"]["prompt_tokens"] += prompt_tokens
        model_usage["coordinator"]["completion_tokens"] += completion_tokens
        model_usage["coordinator"]["image_count"] += image_count
        
        # Parse JSON response or handle direct bash execution
        try:
            # Check if response contains bash code block - execute directly
            if "```bash" in response_text:
                logger.info("Detected bash code block, executing directly")
                code_start = response_text.find("```bash") + 7
                code_end = response_text.find("```", code_start)
                bash_code = response_text[code_start:code_end].strip()
                
                # Execute bash code immediately
                operation_count = state["operation_count"] + 1
                logger.info(f"\n{'='*80}")
                logger.info(f"[Direct Bash Execution] Operation #{operation_count}")
                logger.info(f"Code:\n{bash_code}")
                logger.info("="*80)
                
                exitcode = 1
                logs = ""
                try:
                    output_dict = state["env"].controller.run_bash_script(bash_code, timeout=300)
                    exitcode = 0 if output_dict["status"] == "success" else 1
                    logs = output_dict["output"]
                except Exception as e:
                    exitcode = 1
                    logs = f"Execution error: {str(e)}"
                    logger.error(traceback.format_exc())
                
                success = (exitcode == 0)
                
                # Take screenshot after execution
                screenshot = state["env"].controller.get_screenshot()
                screenshot_filename = f"step_{operation_count}_bash_direct.png"
                
                with open(os.path.join(state["operations_dir"], screenshot_filename), "wb") as f:
                    f.write(screenshot)
                
                # Log the action
                action_log = {
                    "step": operation_count,
                    "type": "bash_execution_direct",
                    "code": bash_code,
                    "exitcode": exitcode,
                    "output": logs,
                    "cost": 0.0,
                    "screenshot": screenshot_filename
                }
                
                result_message = f"Bash execution {'succeeded' if success else 'failed'}.\nOutput:\n{logs}"
                logger.info(f"Exit code: {exitcode}")
                logger.info(f"Output: {logs[:500]}")
                
                # Add to conversation history
                conversation_history = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response_text}
                ]
                
                # Return state update to continue workflow
                return {
                    "operation_count": operation_count,
                    "current_screenshot": screenshot,
                    "last_execution_result": result_message,
                    "last_execution_success": success,
                    "conversation_history": conversation_history,
                    "action_logs": [action_log],
                    "model_usage": model_usage,
                    "next_node": "coordinator",
                    "task_completed": False,
                    "task_infeasible": False
                }
            
            # Extract JSON content if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            decision = json.loads(repair_json(response_text))
        except json.JSONDecodeError:
            # If JSON parsing fails, try to infer intent from response
            logger.error(f"Failed to parse JSON from response: {response_text}")
            decision = {
                "thought": response_text,
                "action": "terminate",
                "content": response_text
            }
        
        logger.info(f"Thought: {decision.get('thought', 'N/A')[:200]}")
        logger.info(f"Action: {decision.get('action', 'N/A')}")
        logger.info(f"Content preview: {str(decision.get('content', ''))[:100]}")
        
        # Check GUI result if this is feedback after a GUI operation
        gui_success = decision.get('gui_success')
        gui_result_check = decision.get('gui_result_check', '')
        
        if gui_success is False:
            logger.info(f"GUI Result Check: {gui_result_check[:150]}")
        
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
                "conversation_history": conversation_history,
                "model_usage": model_usage,
                "next_node": "evaluator",
                "task_completed": True,
                "task_infeasible": task_infeasible
            }
        elif action == "code":
            logger.info("Next: Code Execution (Bash)")
            return {
                "coordinator_thought": decision.get("thought", ""),
                "coordinator_action": action,
                "coordinator_content": decision.get("content", ""),
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
                "last_gui_success": gui_success,
                "last_gui_result_check": gui_result_check,
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
    """Execute bash commands in the desktop environment."""
    logger = logging.getLogger("desktopenv")
    
    bash_code = state["coordinator_content"]
    env = state["env"]
    operation_count = state["operation_count"] + 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[Bash Execution] Operation #{operation_count}")
    logger.info(f"Code:\n{bash_code}")
    logger.info("="*80)
    
    exitcode = 1
    logs = ""
    
    try:
        output_dict = env.controller.run_bash_script(bash_code, timeout=300)
        exitcode = 0 if output_dict["status"] == "success" else 1
        logs = output_dict["output"]
    except Exception as e:
        exitcode = 1
        logs = f"Execution error: {str(e)}"
        logger.error(traceback.format_exc())
    
    success = (exitcode == 0)
    
    # Take screenshot after execution
    screenshot = env.controller.get_screenshot()
    screenshot_filename = f"step_{operation_count}_bash.png"
    
    with open(os.path.join(state["operations_dir"], screenshot_filename), "wb") as f:
        f.write(screenshot)
    
    # Log the action
    action_log = {
        "step": operation_count,
        "type": "bash_execution",
        "code": bash_code,
        "exitcode": exitcode,
        "output": logs,
        "cost": 0.0,
        "screenshot": screenshot_filename
    }
    
    result_message = f"Bash execution {'succeeded' if success else 'failed'}.\nOutput:\n{logs}"
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
    """Execute GUI operation using computer use API."""
    logger = logging.getLogger("desktopenv")
    
    task = state["coordinator_content"]
    env = state["env"]
    operation_count = state["operation_count"] + 1
    operator_model = state.get("operator_model", "computer-use-preview")
    
    # Check if this is a retry with failure info
    last_gui_success = state.get("last_gui_success")
    last_gui_result_check = state.get("last_gui_result_check", "")
    last_gui_task = state.get("last_gui_task", "")
    last_gui_command = state.get("last_gui_command", "")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[GUI Operation] Operation #{operation_count}")
    logger.info(f"Task: {task}")
    
    # Build previous attempt info
    previous_attempt_info = ""
    if last_gui_success is False and last_gui_result_check:
        previous_attempt_info = f"""
# IMPORTANT - Previous Attempt Failed
Task: {last_gui_task}
Your previous command: {last_gui_command}
Result: FAILED - {last_gui_result_check}

The coordinator verified from the screenshot that your previous action did not achieve the expected result.
Please re-analyze the current screenshot carefully and adjust your coordinates/approach.
"""
        logger.info(f"Previous attempt failed: {last_gui_result_check[:150]}")
    
    logger.info("="*80)
    
    screenshot = state["current_screenshot"]
    screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
    
    # Save screenshot before operation
    screenshot_filename = f"step_{operation_count}_gui.png"
    with open(os.path.join(state["operations_dir"], screenshot_filename), "wb") as f:
        f.write(screenshot)
    
    # Prepare operator prompt
    system_prompt = OPERATOR_SYSTEM_MESSAGE.format(
        CLIENT_PASSWORD=state.get("client_password", "password"),
        previous_attempt_info=previous_attempt_info
    )
    
    history_inputs = [{
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Task: {task}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
        ],
    }]
    
    try:
        client = AbstractLLM(operator_model, logger=logger)
        py_cmd, reasoning = client.call_cua(history_inputs, screen_width=state.get("screen_width", 1920), screen_height=state.get("screen_height", 1080), environment="linux")
        cost, input_tokens, output_tokens, image_count = client.get_usage()

        executed_action = False
        execution_error = None
        if py_cmd:
            try:
                logger.info(f"[GUI command] {py_cmd}")
                obs, *_ = env.step(py_cmd, state.get("sleep_after_execution", 0.5))
                executed_action = True
            except Exception as e:
                logger.error(f"GUI operation error: {e}")
                logger.error(traceback.format_exc())
                execution_error = str(e)
        
        if not reasoning:
            reasoning = "Executed GUI action" if executed_action else "No action executed"

        # Update model usage
        model_usage = state.get("model_usage", {}).copy()
        if "operator" not in model_usage:
            model_usage["operator"] = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "image_count": 0}
        model_usage["operator"]["cost"] += cost
        model_usage["operator"]["prompt_tokens"] += input_tokens
        model_usage["operator"]["completion_tokens"] += output_tokens
        model_usage["operator"]["image_count"] += image_count

        # Get updated screenshot
        screenshot = env.controller.get_screenshot()
        
        action_log = {
            "step": operation_count,
            "type": "gui_operator",
            "task": task,
            "command": py_cmd if py_cmd else "None",
            "result": reasoning,
            "execution_success": executed_action,
            "had_previous_failure": last_gui_success is False,
            "cost": cost,
            "model": state.get("operator_model", "computer-use-preview"),
            "screenshot": screenshot_filename
        }
        
        result_message = f"GUI operation executed: {reasoning}\nCommand: {py_cmd if py_cmd else 'None'}\n\nCoordinator: Please check the screenshot to verify if the intended action succeeded."
        
        if not executed_action and execution_error:
            result_message += f"\n\nExecution error occurred: {execution_error}"
        
        logger.info(f"Execution status: {executed_action}")
        
        return {
            "operation_count": operation_count,
            "current_screenshot": screenshot,
            "last_execution_result": result_message,
            "last_execution_success": executed_action,
            "last_gui_task": task,
            "last_gui_command": py_cmd if py_cmd else "None",
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
            "command": "Error",
            "result": f"Error: {str(e)}",
            "execution_success": False,
            "had_previous_failure": last_gui_success is False,
            "cost": 0.0,
            "screenshot": screenshot_filename
        }
        
        return {
            "operation_count": operation_count,
            "last_execution_result": f"GUI operation failed with exception: {str(e)}",
            "last_execution_success": False,
            "last_gui_task": task,
            "last_gui_command": "Error",
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
    coding_steps = len([log for log in state["action_logs"] if log["type"] == "bash_execution"])
    gui_execution_failures = len([log for log in state["action_logs"] if log["type"] == "gui_operator" and not log.get("execution_success", True)])
    gui_retries = len([log for log in state["action_logs"] if log["type"] == "gui_operator" and log.get("had_previous_failure", False)])
    
    model_usage = state.get("model_usage", {})
    prompt_tokens = sum(model_usage[model]["prompt_tokens"] for model in model_usage)
    completion_tokens = sum(model_usage[model]["completion_tokens"] for model in model_usage)
    total_cost = sum(model_usage[model]["cost"] for model in model_usage)
    
    logger.info(f"Score: {score}")
    logger.info(f"Total operations: {state['operation_count']} (GUI: {cua_steps}, Bash: {coding_steps})")
    logger.info(f"GUI execution failures: {gui_execution_failures}/{cua_steps}")
    logger.info(f"GUI retries after coordinator feedback: {gui_retries}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion")
    
    # Create execution log
    execution_log = {
        "statistics": {
            "score": score,
            "total_steps": state["operation_count"],
            "cua_steps": cua_steps,
            "coding_steps": coding_steps,
            "gui_execution_failures": gui_execution_failures,
            "gui_retries": gui_retries,
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
    """LangGraph-based dual agent framework with direct GUI feedback."""
    
    def __init__(
        self,
        coordinator_model: str = "gpt-o4-mini",
        operator_model: str = "computer-use-preview",
        operator_client_password: str = "",
        screen_width: int = 1920,
        screen_height: int = 1080,
        sleep_after_execution: float = 0.5,
        llm_config_path: str = None,
        max_steps: int = 15,
        history_save_dir: str = "",
        max_gui_subtask_attempts: int = 3
    ):
        """
        Initialize the framework.
        
        Args:
            coordinator_model: Model for high-level planning
            operator_model: Model for GUI operations
            operator_client_password: Password for sudo operations
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            sleep_after_execution: Sleep time after each GUI action
            llm_config_path: Path to LLM configuration
            max_steps: Maximum number of operations
            history_save_dir: Directory to save execution history
            max_gui_subtask_attempts: Maximum number of attempts for a GUI subtask
        """
        self.coordinator_model = coordinator_model
        self.operator_model = operator_model
        self.client_password = operator_client_password
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sleep_after_execution = sleep_after_execution
        self.max_steps = max_steps
        self.history_save_dir = history_save_dir
        self.max_gui_subtask_attempts = max_gui_subtask_attempts
        
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
            "coordinator_gui_feedback": "",
            "last_execution_result": "",
            "last_execution_success": True,
            "current_gui_subtask": "",
            "gui_subtask_attempts": 0,
            "gui_error_history": [],
            "consecutive_gui_failures": 0,
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
            "max_steps": self.max_steps,
            "max_gui_subtask_attempts": self.max_gui_subtask_attempts
        }
        
        logger.info("\n" + "="*80)
        logger.info("Starting LangGraph Workflow (Smart GUI Feedback)")
        logger.info("="*80)
        logger.info(f"Task: {task_config['instruction'][:100]}...")
        logger.info(f"Coordinator model: {self.coordinator_model}")
        logger.info(f"Operator model: {self.operator_model}")
        logger.info(f"Max steps: {self.max_steps}")
        logger.info(f"Max GUI subtask attempts: {self.max_gui_subtask_attempts}")
        logger.info("="*80 + "\n")
        
        # Execute workflow with detailed logging
        try:
            iteration = 0
            for state_update in self.workflow.stream(initial_state, config={"recursion_limit": self.max_steps*2}):
                iteration += 1
                node_name = list(state_update.keys())[0]
                node_state = state_update[node_name]
                
                logger.debug(f"\n{'#'*80}")
                logger.debug(f"Iteration {iteration}: Node '{node_name}' completed")
                logger.debug(f"  Operation count: {node_state.get('operation_count', 0)}")
                logger.debug(f"  Next node: {node_state.get('next_node', 'N/A')}")
                logger.debug(f"  Task completed: {node_state.get('task_completed', False)}")
                logger.debug(f"  Coordinator action: {node_state.get('coordinator_action', 'N/A')}")
                logger.debug(f"  Consecutive GUI failures: {node_state.get('consecutive_gui_failures', 0)}")
                logger.debug(f"  GUI subtask attempts: {node_state.get('gui_subtask_attempts', 0)}")
                logger.debug(f"{'#'*80}\n")
                
                # Stop if we've reached the end
                if node_name == "evaluator":
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
                    "gui_execution_failures": 0,
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
                "gui_error_history": [],
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