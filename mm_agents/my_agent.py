#!/usr/bin/env python3
import base64
import glob
from typing import List, Optional, Tuple
import json
import os
import logging
from configs.config import OPENAI_API_KEY
from pydantic import SecretStr
from mm_agents.coact.autogen import LLMConfig
from desktop_env.envs.desktop_env import DesktopEnv
from mm_agents.coact.autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from mm_agents.coact.autogen.code_utils import PYTHON_VARIANTS
from mm_agents.coact.cua_agent import call_openai_cua, _cua_to_pyautogui
from openai import OpenAI
from utils import serialize_json


# ==================== COORDINATOR AGENT ====================

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
"""

OPERATOR_SYSTEM_MESSAGE = """# Task
{instruction}

# Hints
- Sudo password is "{CLIENT_PASSWORD}".
- Execute only the specific GUI action requested.
- Use precise coordinates for clicking and typing.
"""


class CoordinatorAgent(MultimodalConversableAgent):
    """
    Coordinator Agent: Handles task analysis, planning, and code execution.
    Delegates GUI operations to Operator when necessary.
    """
    
    CALL_OPERATOR_TOOL = {
        "type": "function",
        "function": {
            "name": "call_operator",
            "description": "Let a GUI Operator to execute ONE specific GUI action. Call multiple times for complex tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "[REQUIRED] Specific GUI action to perform (e.g., 'Click the Save button', 'Type username in the login field').",
                    },
                    "context": {
                        "type": "string", 
                        "description": "[REQUIRED] Current situation and what you want to achieve with this action.",
                    }
                },
                "required": ["task", "context"]
            },
        },
    }
    
    def __init__(
        self,
        name: str = "coordinator",
        system_message: str = COORDINATOR_SYSTEM_MESSAGE,
        **kwargs
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            **kwargs
        )
        # Register the operator calling tool
        self.update_tool_signature(self.CALL_OPERATOR_TOOL, is_remove=False)


class OperatorAgent:
    """
    Operator Agent: Handles GUI operations using OpenAI's computer use API.
    Receives high-level instructions from Coordinator and executes precise GUI actions.
    """
    
    def __init__(
        self,
        client_password: str = "",
        screen_width: int = 1920,
        screen_height: int = 1080,
        sleep_after_execution: float = 0.5,
        operator_model: str = "computer-use-preview"
    ):
        self.client = OpenAI()
        self.client_password = client_password
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sleep_after_execution = sleep_after_execution
        self.operator_model = operator_model
        
    def execute_single_action(
        self, 
        env: DesktopEnv,
        task: str, 
        context: str,
        save_path: str = "./",
        step_number: int = 1
    ) -> Tuple[str, float]:
        """
        Execute a single GUI action using OpenAI's computer use API.
        
        Args:
            env: Desktop environment
            task: High-level task description from Coordinator
            context: Task context and expected outcome
            save_path: Path to save screenshots and logs
            step_number: Current step number for logging
            
        Returns:
            Tuple of (result_message, cost)
        """
        logger = logging.getLogger("desktopenv")
        
        # Get current screenshot
        obs = env.controller.get_screenshot()
        screenshot_b64 = base64.b64encode(obs).decode("utf-8")
        
        # Save screenshot with unified naming
        with open(os.path.join(save_path, f"step_{step_number}_gui.png"), "wb") as f:
            f.write(obs)
        
        # Prepare prompt for single action
        task_instruction = f"""# GUI Action Request
{task}

# Context
{context}

# Instructions
Analyze the current screenshot and execute the requested GUI operation."""
        
        operator_prompt = OPERATOR_SYSTEM_MESSAGE.format(
            instruction=task_instruction,
            CLIENT_PASSWORD=self.client_password
        )
        
        # Prepare API call
        history_inputs = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": operator_prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"},
            ],
        }]
        
        try:
            # Call OpenAI computer use API for single action
            response, cost, input_tokens, output_tokens = call_openai_cua(
                self.client, 
                history_inputs, 
                self.screen_width, 
                self.screen_height,
                model=self.operator_model
            )
            
            # Process response
            reasoning = ""
            executed_action = False
            
            for output_item in response.output:
                output_type = output_item.get("type", "") if isinstance(output_item, dict) else getattr(output_item, "type", "")
                
                if "computer_call" in str(output_type):
                    # Execute the GUI action
                    action_call = output_item if isinstance(output_item, dict) else output_item.model_dump()
                    py_cmd = _cua_to_pyautogui(action_call["action"])
                    
                    # Execute in environment
                    obs, *_ = env.step(py_cmd, self.sleep_after_execution)
                    executed_action = True
                    
                    logger.info(f"[Operator Step {step_number}]: Executed {py_cmd}")
                    
                elif "reasoning" in str(output_type) and hasattr(output_item, 'summary') and len(output_item.summary) > 0:
                    reasoning = output_item.summary[0].text
                    logger.info(f"[Operator Reasoning]: {reasoning}")
                    
                elif "message" in str(output_type):
                    message_text = output_item.content[0].text if hasattr(output_item, 'content') else str(output_item)
                    reasoning = message_text
            
            if not reasoning:
                if executed_action:
                    reasoning = f"Executed GUI action at step {step_number}"
                else:
                    reasoning = f"No action executed at step {step_number}"
                    
        except Exception as e:
            reasoning = f"Error during GUI operation: {str(e)}"
            cost = 0.0
            input_tokens = 0
            output_tokens = 0
            logger.error(f"Operator error: {e}")
        
        return reasoning, cost, input_tokens, output_tokens


class MyAgentFramework:
    """
    Main framework orchestrating Coordinator and Operator agents.
    Simplified two-agent architecture for task execution.
    """
    
    def __init__(
        self,
        coordinator_model: str = "o3-2025-04-16",
        operator_client_password: str = "",
        operator_model: str = "computer-use-preview",
        screen_width: int = 1920,
        screen_height: int = 1080,
        sleep_after_execution: float = 0.5,
        llm_config_path: str = "mm_agents/coact/OAI_CONFIG_LIST",
        max_steps: int = 15,
        history_save_dir: str = "./results/dual_agent"
    ):
        # Initialize LLM config
        self.llm_config = LLMConfig.from_json(path=llm_config_path).where(model=coordinator_model)
        
        # Update API keys
        for config_item in self.llm_config.config_list:
            config_item.api_key = SecretStr(OPENAI_API_KEY)
        
        # Initialize agents
        self.coordinator = CoordinatorAgent(
            llm_config=self.llm_config,
            system_message=COORDINATOR_SYSTEM_MESSAGE
        )
        
        self.operator = OperatorAgent(
            client_password=operator_client_password,
            operator_model=operator_model,
            screen_width=screen_width,
            screen_height=screen_height,
            sleep_after_execution=sleep_after_execution
        )
        
        # Configuration
        self.max_steps = max_steps
        self.history_save_dir = history_save_dir
        self.total_steps = 0
        self.action_logs = []  # Unified action log list
        self.coordinator_model = coordinator_model
        self.operator_model = operator_model
        
        # Track usage by model
        self.model_usage = {}
        
        # Environment will be set during task execution
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
    ) -> Tuple[float, List[dict]]:
        """
        Execute a task using the dual agent framework.
        
        Args:
            task_config: Task configuration containing instruction and other details
            additional_context: Optional additional context for task assistance
            
        Returns:
            Tuple of (task_score, chat_history)
        """
        if not self.env:
            raise ValueError("Environment not initialized. Call setup_environment() first.")
        
        # Reset environment
        self.env.reset(task_config=task_config)
        
        # Get initial screenshot (for message only, not saved)
        screenshot = self.env.controller.get_screenshot()
        
        # Prepare initial message
        initial_message = task_config["instruction"] + additional_context + '\n\nCheck my computer screenshot and describe it first. If this task is possible to complete, please complete it on my computer. If not, reply with "INFEASIBLE" to end the conversation.\nI will not provide further information to you.'
        
        # Add screenshot
        screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
        initial_message += f"\n\n<img data:image/png;base64,{screenshot_b64}>"
        
        # Create a simple proxy agent for conversation
        proxy_agent = MultimodalConversableAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            llm_config=False,
            code_execution_config={
                "use_docker": False,
                "timeout": 300,
                "last_n_messages": 1,
            },
            is_termination_msg=lambda x: (
                x.get("content", "") and 
                any(
                    ("terminate" in content.get("text", "").lower() or 
                     "infeasible" in content.get("text", "").lower())
                    for content in x.get("content", [])
                    if isinstance(content, dict) and content.get("type") == "text"
                )
            ),
            max_consecutive_auto_reply=self.max_steps,
            default_auto_reply="Continue with the task. If completed, reply with 'TERMINATE'. If impossible, reply with 'INFEASIBLE'."
        )
        
        # Enable code execution for coordinator
        def run_code(code: str, lang: str = "python", **kwargs):
            """Execute code in the desktop environment."""
            exitcode = 1
            logs = ""
            
            try:
                if lang in ["bash", "shell", "sh"]:
                    output_dict = self.env.controller.run_bash_script(code, timeout=300)
                    exitcode = 0 if output_dict["status"] == "success" else 1
                    logs = output_dict["output"]
                elif lang in PYTHON_VARIANTS:
                    output_dict = self.env.controller.run_python_script(code, timeout=300)
                    exitcode = 0 if output_dict["status"] != "error" else 1
                    logs = output_dict.get("message", output_dict.get("output", ""))
                else:
                    exitcode = 1
                    logs = f"Unsupported language: {lang}"
            except Exception as e:
                exitcode = 1
                logs = f"Execution error: {str(e)}"
            
            # Update step count and log the coding action
            self.total_steps += 1
            
            # Take screenshot after code execution
            screenshot = self.env.controller.get_screenshot()
            screenshot_filename = f"step_{self.total_steps}_code.png"
            with open(os.path.join(operations_dir, screenshot_filename), "wb") as f:
                f.write(screenshot)
            
            # Log the coding action
            action_log = {
                "step": self.total_steps,
                "type": "code_execution",
                "language": lang,
                "code": code,
                "exitcode": exitcode,
                "output": logs,
                "cost": 0.0,  # No cost for code execution
                "screenshot": screenshot_filename
            }
            self.action_logs.append(action_log)
            
            return exitcode, logs, None
        
        # Monkey patch the run_code method
        proxy_agent.run_code = run_code
        
        # Create unified operations directory
        operations_dir = os.path.join(self.history_save_dir, "operations")
        os.makedirs(operations_dir, exist_ok=True)
        
        # Register functions on proxy_agent (not coordinator)
        def call_operator(task: str, context: str) -> str:
            """Call the operator to perform a single GUI operation."""
            
            # Update step count first
            self.total_steps += 1
            
            # Execute single GUI action
            result, cost, input_tokens, output_tokens = self.operator.execute_single_action(
                env=self.env,
                task=task,
                context=context,
                save_path=operations_dir,
                step_number=self.total_steps
            )
            
            # Log the action to unified log
            action_log = {
                "step": self.total_steps,
                "type": "gui_operator",
                "task": task,
                "context": context,
                "result": result,
                "cost": cost,
                "model": self.operator_model,
                "screenshot": f"step_{self.total_steps}_gui.png"
            }
            self.action_logs.append(action_log)
            
            # Update model-specific usage
            if "operator" not in self.model_usage:
                self.model_usage["operator"] = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
            self.model_usage["operator"]["cost"] += cost
            self.model_usage["operator"]["prompt_tokens"] += input_tokens
            self.model_usage["operator"]["completion_tokens"] += output_tokens
            
            # Get current screenshot after action
            screenshot = self.env.controller.get_screenshot()
            screenshot_b64 = base64.b64encode(screenshot).decode("utf-8")
            
            return f"# GUI Operator Result\n{result}\n\n<img data:image/png;base64,{screenshot_b64}>"
        
        # Register the function on proxy_agent
        proxy_agent.register_function(
            function_map={"call_operator": call_operator}
        )
        
        # Start conversation
        with self.llm_config:
            proxy_agent.initiate_chat(
                recipient=self.coordinator,
                message=initial_message,
                max_turns=self.max_steps
            )
        
        # Extract chat history
        chat_history = []
        if proxy_agent.chat_messages:
            key = list(proxy_agent.chat_messages.keys())[0]
            chat_messages = proxy_agent.chat_messages[key]
            
            for item in chat_messages:
                # Clean up the message for storage
                cleaned_item = item.copy()
                cleaned_item.pop('tool_responses', None)
                
                # Replace image URLs with placeholder
                if cleaned_item.get('content'):
                    for content in cleaned_item['content']:
                        if isinstance(content, dict) and content.get('type') == 'image_url':
                            content['image_url'] = "<image>"
                
                chat_history.append(cleaned_item)
        
        # Get token usage from coordinator
        coordinator_usage = self.coordinator.get_total_usage()
        coordinator_prompt_tokens = coordinator_usage.get('prompt_tokens', 0)
        coordinator_completion_tokens = coordinator_usage.get('completion_tokens', 0)
        coordinator_cost = coordinator_usage.get('cost', 0.0)
        if "coordinator" not in self.model_usage:
            self.model_usage["coordinator"] = {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
        self.model_usage["coordinator"]["prompt_tokens"] += coordinator_prompt_tokens
        self.model_usage["coordinator"]["completion_tokens"] += coordinator_completion_tokens
        self.model_usage["coordinator"]["cost"] += coordinator_cost
        
        prompt_tokens = sum(self.model_usage[model]["prompt_tokens"] for model in self.model_usage)
        completion_tokens = sum(self.model_usage[model]["completion_tokens"] for model in self.model_usage)
        total_cost = sum(self.model_usage[model]["cost"] for model in self.model_usage)
        cua_steps = len([log for log in self.action_logs if log["type"] == "gui_operator"])
        code_operations = len([log for log in self.action_logs if log["type"] == "code_execution"])

        # Evaluate task completion
        try:
            score = self.env.evaluate()
        except Exception as e:
            logging.getLogger("desktopenv").error(f"Evaluation error: {e}")
            score = 0.0
        
        print(f"Score: {score}")
        print(f"Total operations: {cua_steps + code_operations} (GUI: {cua_steps}, Code: {code_operations})")
        print(f"Total cost: ${total_cost:.4f}")

        # Create unified execution log combining chat history and actions
        unified_log = {
            "statistics": {
                "score": score,
                "total_steps": self.total_steps,
                "cua_steps": cua_steps,
                "coding_steps": code_operations,
                "total_cost": total_cost,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model_usage": self.model_usage
            },
            "task_config": task_config,
            "additional_context": additional_context,
            "chat_history": chat_history,
            "action_logs": self.action_logs
        }
        
        # Save unified execution log
        with open(os.path.join(self.history_save_dir, "execution_log.json"), "w") as f:
            json.dump(serialize_json(unified_log), f, indent=2)

        return score
    
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            self.env.close()
            self.env = None