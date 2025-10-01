#!/usr/bin/env python3
"""
LangGraph-based Dual Agent Framework
Complete implementation with bash-only execution and proper state management
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

COORDINATOR_SYSTEM_MESSAGE = """# Your role
You are a task solver, you need to complete a computer-using task step-by-step.
1. Provide a detailed plan, including a list of user requirements like specific file name, file path, etc.
2. Follow the following instructions and complete the task with your skills.
    - If you think the task is impossible to complete (no file, wrong environment, etc.), reply with "INFEASIBLE" to end the conversation.
    - **Do not** do anything else out of the user's instruction like change the file name. This will make the task fail.
    - Check every screenshot carefully and see if it fulfills the task requirement.
    - You MUST try code execution first for file operation tasks like spreadsheet modification.
3. Verify the result and see if it fulfills the user's requirement.

# Your skills
You can use the following tools to solve the task:

## Code Execution (Bash Only)
You can write bash commands in ```bash...``` code blocks. ALL code execution must be in bash format.
- Your linux username is "user"
- If you want to use sudo, follow the format: "echo {CLIENT_PASSWORD} | sudo -S [YOUR COMMANDS]" (no quotes for the word "{CLIENT_PASSWORD}")
- You MUST verify the result before save the changes
- Wrap all your code in ONE bash code block. DO NOT let user save the code as a file and execute it for you
- When you modify a spreadsheet, **make sure every value is in the expected cell**
- You need to print the progressive and final result

### Python Execution via Bash
If you need to run Python code, use bash format with inline Python.
- Install packages if needed, for example: `pip install numpy && python3 -c "import numpy; print('success')"`
- Always check if packages are installed before using them

## GUI Operator
Let a GUI Operator to solve a subtask you assigned.
GUI Operator can operate the computer by clicking and typing (but not accurate).
Require a detailed task description.
The Operator executes ONE GUI action per call (one click, one type, etc.). Each step is a one-time interaction with OS like mouse click or keyboard typing. Please take this into account when you plan the actions.
If you let GUI Operator to check the result, you MUST let it close and reopen the file because programmer's result will NOT be updated to the screen.

Remember: Use bash execution (including inline Python) for file operations and data processing. Use GUI Operator for visual interactions that require precise clicking or typing.

# Decision Format
Your response MUST be in JSON format:
{
    "thought": "Your reasoning about current situation and next action",
    "action": "code|gui|terminate",
    "content": "Bash commands to execute OR task description for GUI operator"
}

If task is completed successfully, use action "terminate" with thought explaining completion.
If task is impossible, use action "terminate" with thought "INFEASIBLE: [reason]".

IMPORTANT: For "code" action, always provide bash commands. Never use Python directly.
"""

CUA_SYSTEM_MESSAGE = """# Task
{instruction}

# Hints
- Sudo password is "{CLIENT_PASSWORD}".
- Execute only the specific GUI action requested.
- Use precise coordinates for clicking and typing.
"""

OPERATOR_SYSTEM_MESSAGE = """You are an agent which follow my instruction and perform desktop computer tasks as instructed.
You have good knowledge of computer and good internet connection and assume your code will run on a computer for controlling the mouse and keyboard.
You will get an observation of an image, which is the screenshot of the computer screen and you will predict the action of the computer based on the image.

You are required to use `pyautogui` to perform the action grounded to the observation, but DONOT use the `pyautogui.locateCenterOnScreen` function to locate the element you want to operate with since we have no image of the element you want to operate with. DONOT USE `pyautogui.screenshot()` to make screenshot.
Return one line or multiple lines of python code to perform the action each time, be time efficient. When predicting multiple lines of code, make some small sleep like `time.sleep(0.5);` interval so that the machine could take; Each time you need to predict a complete code, no variables or function can be shared from history
You need to to specify the coordinates of by yourself based on your observation of current observation, but you should be careful to ensure that the coordinates are correct.
You ONLY need to return the code inside a code block, like this:
```python
# your code here
```

My computer's password is '{CLIENT_PASSWORD}', feel free to use it when you need sudo rights.
First give the current screenshot and previous things we did a short reflection, then RETURN ME THE CODE. NEVER EVER RETURN ME ANYTHING ELSE."""


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

    client = AbstractLLM(state.get("coordinator_model"))
    
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
        
        # Parse JSON response
        try:
            print("response_text:\n", response_text)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            decision = json.loads(repair_json(response_text))
        except json.JSONDecodeError:
            # Fallback: treat as terminate action   
            decision = {
                "thought": response_text,
                "action": "terminate",
                "content": response_text
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
        # Always execute as bash script
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
    """Execute GUI operation using OpenAI's computer use API."""
    logger = logging.getLogger("desktopenv")
    
    task = state["coordinator_content"]
    context = state["coordinator_thought"]
    env = state["env"]
    operation_count = state["operation_count"] + 1
    operator_model = state.get("operator_model", "computer-use-preview")
    
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
    
    if operator_model == "computer-use-preview":
        operator_prompt = CUA_SYSTEM_MESSAGE.format(
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
    else:
        system_prompt = OPERATOR_SYSTEM_MESSAGE.format(
            CLIENT_PASSWORD=state.get("client_password", "password")
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
                    {"type": "text", "text": task_instruction},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{screenshot_b64}"},
                ],
            }
        ]
    
    try:
        client = AbstractLLM(operator_model)
        py_cmd, reasoning = client.call_cua(history_inputs, screen_width=state.get("screen_width", 1920), screen_height=state.get("screen_height", 1080), environment="linux")
        cost, input_tokens, output_tokens, image_count = client.get_usage()

        executed_action = False
        if py_cmd:
            try:
                logger.info(f"[GUI command] {py_cmd}")
                obs, *_ = env.step(py_cmd, state.get("sleep_after_execution", 0.5))
                executed_action = True
            except Exception as e:
                logger.error(f"GUI operation error: {e}")
                logger.error(traceback.format_exc())
        
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
    bash_steps = len([log for log in state["action_logs"] if log["type"] == "bash_execution"])
    
    model_usage = state.get("model_usage", {})
    prompt_tokens = sum(model_usage[model]["prompt_tokens"] for model in model_usage)
    completion_tokens = sum(model_usage[model]["completion_tokens"] for model in model_usage)
    total_cost = sum(model_usage[model]["cost"] for model in model_usage)
    
    logger.info(f"Score: {score}")
    logger.info(f"Total operations: {state['operation_count']} (GUI: {cua_steps}, Bash: {bash_steps})")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion")
    
    # Create execution log
    execution_log = {
        "statistics": {
            "score": score,
            "total_steps": state["operation_count"],
            "cua_steps": cua_steps,
            "coding_steps": bash_steps,
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
    """LangGraph-based dual agent framework with bash-only execution."""
    
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
        logger.info("Starting LangGraph Workflow Execution (Bash-only)")
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
                    "bash_steps": 0,
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