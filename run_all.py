import argparse
import os
import sys
import signal
import time
import json
from multiprocessing import Process, Manager, current_process
from typing import List

sys.path.append("../GUIAgent")
from utils import summary, setup_logger

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False
logger = None  # Will be initialized in run()

# Load environment variables from .env file
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()


class SimpleVMPool:
    """Simple VM pool manager for multiprocessing without external dependencies"""
    def __init__(self, vm_data_dir="./vm_data", registry_file=".vms"):
        self.vm_data_dir = vm_data_dir
        self.registry_file = registry_file
        self._init_registry()
    
    def _init_registry(self):
        """Initialize registry file if it doesn't exist"""
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, 'w') as f:
                f.write("")
    
    def _find_available_vms(self):
        """Find all available VM paths in vm_data directory"""
        vms = []
        if not os.path.exists(self.vm_data_dir):
            return vms
        
        for vm_name in os.listdir(self.vm_data_dir):
            vm_path = os.path.join(self.vm_data_dir, vm_name)
            if os.path.isdir(vm_path) and not vm_name.endswith('.zip'):
                # Check for .vmx file
                vmx_path = os.path.join(vm_path, vm_name, f"{vm_name}.vmx")
                if os.path.exists(vmx_path):
                    # Normalize path
                    normalized_path = os.path.normpath(vmx_path)
                    vms.append(normalized_path)
        return sorted(vms)
    
    def _create_new_vm(self, source_vm_path, new_vm_name):
        """Automatically create a new VM from zip file"""
        # Check for Ubuntu.zip (Spider2-V uses this name)
        zip_file = "Ubuntu.zip"
        zip_path = os.path.join(self.vm_data_dir, zip_file)
        
        # Method 1: Try using zip file if it exists (like OSWorld)
        if os.path.exists(zip_path):
            print(f"[INFO] Found {zip_file} at {zip_path}, using it to create {new_vm_name}...")
            from desktop_env.envs import _install_virtual_machine
            try:
                # _install_virtual_machine expects (vm_name, vms_dir, downloaded_file_name)
                # It will look for the file at: vms_dir/downloaded_file_name
                new_vm_path = _install_virtual_machine(new_vm_name, self.vm_data_dir, zip_file)
                print(f"[INFO] Successfully created VM from zip: {new_vm_path}")
                return os.path.normpath(new_vm_path)
            except Exception as e:
                print(f"[ERROR] Failed to create from zip: {e}")
                import traceback
                traceback.print_exc()
                raise  # Don't fallback, if we have zip it should work
        
        # No zip file found
        print(f"[ERROR] Ubuntu.zip not found at {zip_path}")
        print(f"[ERROR] Please ensure you have the VM zip file")
        print(f"[ERROR] Cannot create new VM without zip file (vmrun clone doesn't work with running VMs)")
        raise FileNotFoundError(f"Ubuntu.zip not found at {zip_path}")
    
    def allocate_vm(self, process_id, auto_create=True):
        """Allocate a free VM to a process, with file locking for safety"""
        lock_file = self.registry_file + ".lock"
        
        # Simple file-based locking (create lock file)
        max_retries = 30
        for retry in range(max_retries):
            try:
                # Try to create lock file (exclusive)
                if not os.path.exists(lock_file):
                    with open(lock_file, 'w') as f:
                        f.write(str(os.getpid()))
                    break
                else:
                    time.sleep(0.1)
            except:
                time.sleep(0.1)
        
        try:
            # Read current registry
            registry = {}
            if os.path.exists(self.registry_file):
                try:
                    with open(self.registry_file, 'r') as f:
                        for line in f:
                            if '|' in line:
                                vm_path, pid = line.strip().split('|', 1)
                                # Normalize path to avoid duplicates (./vm_data vs vm_data)
                                normalized_vm_path = os.path.normpath(vm_path)
                                registry[normalized_vm_path] = pid
                except:
                    pass
            
            # Find all available VMs
            available_vms = self._find_available_vms()
            
            if not available_vms:
                raise ValueError("No VMs found in vm_data directory")
            
            # Find a free VM
            for vm_path in available_vms:
                if vm_path not in registry or registry[vm_path] == 'free':
                    # Allocate this VM
                    registry[vm_path] = str(process_id)
                    
                    # Write back registry
                    with open(self.registry_file, 'w') as f:
                        for path, pid in registry.items():
                            f.write(f"{path}|{pid}\n")
                    
                    return vm_path
            
            # All VMs are occupied
            if auto_create:
                # Automatically create a new VM
                print(f"[INFO] All {len(available_vms)} VMs are currently occupied")
                
                # Generate placeholder path and reserve it BEFORE creating
                import re
                existing_nums = []
                for vm_name in os.listdir(self.vm_data_dir if os.path.exists(self.vm_data_dir) else '.'):
                    match = re.match(r'Ubuntu(\d+)', vm_name)
                    if match:
                        existing_nums.append(int(match.group(1)))
                
                new_num = max(existing_nums) + 1 if existing_nums else 1
                new_vm_name = f"Ubuntu{new_num}"
                placeholder_path = os.path.join(self.vm_data_dir, new_vm_name, new_vm_name, f"{new_vm_name}.vmx")
                placeholder_path = os.path.normpath(placeholder_path)
                
                # Reserve this VM in registry IMMEDIATELY (before creating)
                registry[placeholder_path] = str(process_id)
                with open(self.registry_file, 'w') as f:
                    for path, pid in registry.items():
                        f.write(f"{path}|{pid}\n")
                
                print(f"[INFO] Reserved {new_vm_name} for process {process_id}, now creating...")
                
                # Now create the actual VM
                new_vm_path = self._create_new_vm(available_vms[0], new_vm_name)
                
                # Update registry with actual path (should be the same)
                if new_vm_path != placeholder_path:
                    del registry[placeholder_path]
                    registry[new_vm_path] = str(process_id)
                    with open(self.registry_file, 'w') as f:
                        for path, pid in registry.items():
                            f.write(f"{path}|{pid}\n")
                
                return new_vm_path
            else:
                raise RuntimeError(f"All {len(available_vms)} VMs are currently occupied")
            
        finally:
            # Release lock
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except:
                pass
    
    def release_vm(self, vm_path):
        """Release a VM back to the pool"""
        lock_file = self.registry_file + ".lock"
        
        # Wait for lock
        max_retries = 30
        for retry in range(max_retries):
            if not os.path.exists(lock_file):
                try:
                    with open(lock_file, 'w') as f:
                        f.write(str(os.getpid()))
                    break
                except:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
        
        try:
            registry = {}
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    for line in f:
                        if '|' in line:
                            path, pid = line.strip().split('|', 1)
                            # Normalize path
                            normalized_path = os.path.normpath(path)
                            registry[normalized_path] = pid
            
            # Normalize the vm_path to release
            normalized_vm_path = os.path.normpath(vm_path)
            if normalized_vm_path in registry:
                registry[normalized_vm_path] = 'free'
            
            with open(self.registry_file, 'w') as f:
                for path, pid in registry.items():
                    f.write(f"{path}|{pid}\n")
        finally:
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except:
                pass


def signal_handler(signum, frame):
    """Handle termination signals (SIGINT, SIGTERM) to gracefully shutdown environments."""
    global is_terminating, active_environments, processes
    
    # Avoid duplicate handling
    if is_terminating:
        return
    
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    
    # Close all registered environments in the main process
    for env in active_environments:
        try:
            logger.info(f"Closing environment...")
            env.close()
            logger.info(f"Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    
    # Send termination signal to all child processes first
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending termination signal to process {p.name}...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error sending termination signal to process: {e}")
    
    # Allow a short time for processes to handle their own cleanup
    time.sleep(1)
    
    # Forcefully terminate any processes that didn't exit
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully killing process {p.name}...")
                p.kill()
            except Exception as e:
                logger.error(f"Error forcefully killing process: {e}")
    
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def filter_tasks(args, test_all_meta: dict, logger) -> List[tuple]:
    """
    Filter tasks based on rerun/rerun_fail flags.
    
    Returns:
        List of (domain, example_id) tuples to execute
    """
    tasks_to_run = []
    
    for domain in test_all_meta:
        for example_id in test_all_meta[domain]:
            target_dir = os.path.join(args.result_dir, domain, example_id)
            result_path = os.path.join(target_dir, 'result.txt')
            err_reason_path = os.path.join(target_dir, 'err_reason.txt')
            
            should_skip = False
            if not args.rerun and os.path.exists(result_path) and not os.path.exists(err_reason_path):
                try:
                    result = float(open(result_path, 'r').read().strip())
                    # Skip successful tasks, or failed tasks if not rerun_fail
                    if result > 0.0 or not args.rerun_fail:
                        should_skip = True
                        logger.info(f"Skipping {domain}/{example_id}, result: {result}")
                except (ValueError, IOError) as e:
                    logger.warning(f"Failed to read result for {domain}/{example_id}: {e}")
            
            if not should_skip:
                tasks_to_run.append((domain, example_id))
    
    return tasks_to_run


def distribute_tasks(test_all_meta: dict) -> List[tuple]:
    """Distribute tasks from test metadata."""
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks


def run_env_tasks(task_queue, args: argparse.Namespace):
    """
    Run tasks in a separate process with its own environment.
    
    Each process:
    1. Allocates a VM from the pool
    2. Creates its own VM environment
    3. Pulls tasks from the shared queue
    4. Calls the appropriate run_xxx.py function with task_filter parameter
    5. Releases the VM back to the pool
    
    The task_filter parameter ensures each process only executes one task at a time.
    """
    env = None
    allocated_vm_path = None
    
    # Use setup_logger to configure subprocess logger (consistent with main process)    
    result_name = os.path.basename(args.result_dir)
    process_logger = setup_logger(f"{result_name}-{current_process().name}", args.log_level)
    
    # Ensure stdout/stderr are flushed (for print statements from _install_virtual_machine)
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        process_logger.info(f"[DEBUG] {current_process().name} starting initialization...")
        
        # Allocate VM from pool if multiprocessing is enabled
        # Extract vm_data directory from path_to_vm
        # path_to_vm format: ./vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx
        # We need: ./vm_data
        if args.path_to_vm:
            # Go up 3 levels from .vmx file to get vm_data directory
            vm_data_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.path_to_vm)))
            if not vm_data_dir or not os.path.exists(vm_data_dir):
                vm_data_dir = "./vm_data"
        else:
            vm_data_dir = "./vm_data"
        
        if args.num_envs > 1:
            # Multi-process mode: allocate VM from pool (like OSWorld)
            process_logger.info(f"[DEBUG] {current_process().name} using vm_data_dir: {vm_data_dir}")
            vm_pool = SimpleVMPool(vm_data_dir=vm_data_dir)
            try:
                allocated_vm_path = vm_pool.allocate_vm(os.getpid(), auto_create=True)
                process_logger.info(f"[INFO] {current_process().name} allocated VM: {allocated_vm_path}")
            except Exception as e:
                process_logger.error(f"[ERROR] {current_process().name} failed to allocate VM: {e}")
                process_logger.error(f"[ERROR] Available VMs: {vm_pool._find_available_vms()}")
                process_logger.error(f"[ERROR] Please ensure you have at least one VM in vm_data directory")
                return
        else:
            # Single process mode, use the provided VM path
            allocated_vm_path = args.path_to_vm
            process_logger.info(f"[INFO] {current_process().name} using VM: {allocated_vm_path}")
        
        # Import desktop environment
        try:
            from desktop_env.envs.desktop_env import DesktopEnv
        except Exception as e:
            process_logger.error(f"[DEBUG] Failed to import DesktopEnv from desktop_env.envs: {e}")
            from desktop_env.desktop_env import DesktopEnv
        
        # Create environment for this process with allocated VM
        try:
            process_logger.info(f"[DEBUG] {current_process().name} attempting to create DesktopEnv with provider_name...")
            env = DesktopEnv(
                # provider_name=args.provider_name,
                path_to_vm=allocated_vm_path,
                action_space=args.action_space,
                snapshot_name=args.snapshot_name,
                headless=args.headless,
                require_a11y_tree=False,
                # enable_proxy=False,
                screen_size=(args.screen_width, args.screen_height)
            )
            process_logger.info(f"[DEBUG] {current_process().name} DesktopEnv created successfully with provider_name")
        except Exception as e1:
            process_logger.warning(f"[DEBUG] {current_process().name} failed with provider_name: {e1}")
            process_logger.info(f"[DEBUG] {current_process().name} attempting to create DesktopEnv without provider_name...")
            try:
                env = DesktopEnv(
                    # provider_name=args.provider_name,
                    path_to_vm=allocated_vm_path,
                    snapshot_name=args.snapshot_name,
                    action_space=args.action_space,
                    headless=args.headless,
                    require_a11y_tree=False,
                    screen_size=(args.screen_width, args.screen_height)
                )
                process_logger.info(f"[DEBUG] {current_process().name} DesktopEnv created successfully without provider_name")
            except Exception as e2:
                process_logger.error(f"[DEBUG] {current_process().name} failed to create DesktopEnv: {e2}")
                import traceback
                process_logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
                raise
        
        # Set environment in args
        args.env = env
        
        process_logger.info(f"Process {current_process().name} started with method: {args.method}")
        
        # Process tasks from queue
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                # Queue is empty, exit
                break
            
            domain, example_id = item
            process_logger.info(f"[{current_process().name}] Processing {domain}/{example_id}")
            
            try:
                # Prepare task arguments
                import copy
                task_args = copy.deepcopy(args)
                task_args.get_score = False
                
                # Create tasks list with only this task
                single_task = [(domain, example_id)]
                
                # Import and execute the appropriate run function
                if args.method == "spider2v_agent":
                    from run_spider2v_agent import run
                elif args.method == "coact":
                    from run_coact import run
                elif args.method == "agents3":
                    from run_agents3 import run
                elif args.method == "agents2":
                    from run_agents2 import run
                elif args.method == "cmm":
                    from run_cmm import run
                elif args.method == "gta1":
                    from run_gta1_agent import run
                else:
                    raise ValueError(f"Invalid method: {args.method}")
                
                # Execute the task with logger and tasks
                run(task_args, logger=process_logger, tasks=single_task)
                
            except Exception as e:
                process_logger.error(f"Task-level error in {current_process().name} for {domain}/{example_id}: {e}")
                import traceback
                process_logger.error(traceback.format_exc())
                
                # Create result directory if it doesn't exist
                example_result_dir = os.path.join(args.result_dir, domain, example_id)
                os.makedirs(example_result_dir, exist_ok=True)
                
                # Save error information
                with open(os.path.join(example_result_dir, "result.txt"), "w") as f:
                    f.write("0.0")
                with open(os.path.join(example_result_dir, "err_reason.txt"), "w") as f:
                    f.write(f"Fatal error: {str(e)}\n\n{traceback.format_exc()}")
    
    except Exception as e:
        process_logger.error(f"Process-level error in {current_process().name}: {e}")
        import traceback
        process_logger.error(traceback.format_exc())
    
    finally:
        process_logger.info(f"{current_process().name} cleaning up environment...")
        try:
            if env:
                # Try to close environment (VM might already be stopped)
                try:
                    env.close()
                    process_logger.info(f"{current_process().name} environment closed successfully")
                except Exception as close_error:
                    error_msg = str(close_error)
                    if "not powered on" in error_msg or "not running" in error_msg:
                        process_logger.info(f"{current_process().name} VM already stopped, skipping close")
                    else:
                        process_logger.warning(f"{current_process().name} error closing environment: {close_error}")
                
                # Clean VMware lock files to prevent lock issues in multiprocessing
                if allocated_vm_path:
                    vm_dir = os.path.dirname(allocated_vm_path)
                    try:
                        env.clean_lock(vm_dir)
                        process_logger.info(f"{current_process().name} lock files cleaned")
                    except Exception as lock_error:
                        process_logger.debug(f"{current_process().name} error cleaning locks: {lock_error}")
        except Exception as e:
            process_logger.error(f"{current_process().name} unexpected error during cleanup: {e}")
        
        # Release VM back to pool
        if args.num_envs > 1 and allocated_vm_path:
            try:
                # Extract vm_data directory (same logic as above)
                if args.path_to_vm:
                    vm_data_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.path_to_vm)))
                    if not vm_data_dir or not os.path.exists(vm_data_dir):
                        vm_data_dir = "./vm_data"
                else:
                    vm_data_dir = "./vm_data"
                
                vm_pool = SimpleVMPool(vm_data_dir=vm_data_dir)
                vm_pool.release_vm(allocated_vm_path)
                process_logger.info(f"[INFO] {current_process().name} released VM: {allocated_vm_path}")
            except Exception as e:
                process_logger.error(f"[ERROR] {current_process().name} error releasing VM: {e}")


def run_multienv(args: argparse.Namespace, tasks_to_run: List[tuple]):
    """Run tasks with multiple environments in parallel."""
    global processes
    
    logger.info("Args: %s", args)
    logger.info(f"Total tasks: {len(tasks_to_run)}")
    
    # Stop all VMs and reset registry before starting multiprocess
    logger.info("Preparing VMs for multiprocess mode...")
    try:
        import subprocess
        
        # Get list of running VMs
        result = subprocess.run(["vmrun", "-T", "ws", "list"], 
                               capture_output=True, text=True, 
                               encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            running_vms = result.stdout.strip().split('\n')[1:]  # Skip first line "Total running VMs: X"
            
            # Stop each running VM
            for vm_path in running_vms:
                vm_path = vm_path.strip()
                if vm_path and 'vm_data' in vm_path:
                    logger.info(f"Stopping VM: {vm_path}")
                    try:
                        subprocess.run(["vmrun", "-T", "ws", "stop", vm_path],
                                     capture_output=True, timeout=30)
                    except Exception as e:
                        logger.warning(f"Failed to stop {vm_path}: {e}")
            
            time.sleep(2)  # Wait for VMs to fully stop
        
        # Reset .vms registry - mark all VMs as free
        registry_file = ".vms"
        if os.path.exists(registry_file):
            logger.info("Resetting VM registry...")
            new_lines = []
            with open(registry_file, 'r') as f:
                for line in f:
                    if '|' in line:
                        vm_path, _ = line.strip().split('|', 1)
                        new_lines.append(f"{vm_path}|free\n")
            
            with open(registry_file, 'w') as f:
                f.writelines(new_lines)
            
            logger.info("All VMs marked as free")
    
    except Exception as e:
        logger.warning(f"Failed to prepare VMs: {e}")
        logger.warning("Continuing anyway...")
    
    with Manager() as manager:
        task_queue = manager.Queue()
        for item in tasks_to_run:
            task_queue.put(item)
        
        num_envs = args.num_envs
        processes = []
        
        for i in range(num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args),
                name=f"EnvProcess-{i+1}"
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")
        
        try:
            # Monitor processes and restart if needed
            last_queue_size = len(tasks_to_run)
            while True:
                alive_count = 0
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        logger.info(f"Process {p.name} died, restarting...")
                        new_p = Process(
                            target=run_env_tasks,
                            args=(task_queue, args),
                            name=f"EnvProcess-Restart-{idx+1}"
                        )
                        new_p.daemon = True
                        new_p.start()
                        processes[idx] = new_p
                        logger.info(f"Restarted process {new_p.name} with PID {new_p.pid}")
                    else:
                        alive_count += 1
                
                # Show progress
                current_queue_size = task_queue.qsize()
                completed = len(tasks_to_run) - current_queue_size
                if current_queue_size != last_queue_size:
                    logger.info(f"Progress: {completed}/{len(tasks_to_run)} tasks completed, {alive_count} processes running")
                    last_queue_size = current_queue_size
                
                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                
                time.sleep(5)
            
            # Wait for all processes to finish
            for p in processes:
                p.join()
                
        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Initiating graceful shutdown...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while waiting for processes: {e}", exc_info=True)
            for p in processes:
                if p.is_alive():
                    try:
                        logger.info(f"Terminating process {p.name} due to error...")
                        p.terminate()
                    except Exception as term_e:
                        logger.error(f"Error terminating process {p.name}: {term_e}")
            raise


def run():
    parser = argparse.ArgumentParser(description="Run evaluation for agent framework")

    # General config
    parser.add_argument(
        "--method", type=str, default="cmm", help="Method to use"
    )

    # Environment config
    parser.add_argument(
        "--provider_name", type=str, default="vmware", help="Provider name"
    )
    parser.add_argument(
        "--path_to_vm",
        type=str,
        default="./vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx",
        help="Path to VM file",
    )
    parser.add_argument("--snapshot_name", type=str, default="low_res")
    parser.add_argument("--screen_width", type=int, default=1280)
    parser.add_argument("--screen_height", type=int, default=720)
    parser.add_argument("--sleep_after_execution", type=float, default=0.5)
    parser.add_argument(
        "--client_password", type=str, default="password", help="VM client password"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode or machine"
    )
    parser.add_argument("--record", action="store_true", help="Record the execution process")
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )

    # Agent config
    parser.add_argument(
        "--executive_controller_model",
        type=str,
        default="o3",
        help="Model for Executive Controller agent",
    )
    parser.add_argument(
        "--visuomotor_mapper_model",
        type=str,
        default="computer-use-preview",
        help="Model for Visuomotor Mapper agent",
    )
    parser.add_argument("--memory_consolidator_model", type=str, default="gpt-5-mini",
                       help="Model for auxiliary tasks (episodic encoding, periodic consolidation, schema induction, etc.)")
    parser.add_argument(
        "--max_steps", type=int, default=15, help="Maximum steps for Executive Controller"
    )
    parser.add_argument(
        "--oai_config_path",
        type=str,
        default="D:/projects/GUIAgent/coact/OAI_CONFIG_LIST",
    )
    parser.add_argument("--orchestrator_model", type=str, default="o3-2025-04-16")
    parser.add_argument("--coding_model", type=str, default="o4-mini-2025-04-16")
    parser.add_argument("--cua_model", type=str, default="computer-use-preview")
    parser.add_argument("--orchestrator_max_steps", type=int, default=15)
    parser.add_argument("--coding_max_steps", type=int, default=20)
    parser.add_argument("--cua_max_steps", type=int, default=25)
    parser.add_argument("--cut_off_steps", type=int, default=50)
    parser.add_argument("--max_trajectory_length", type=int, default=3)

    # Task/example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default=os.path.join("evaluation_examples", "test_one.json"),
    )
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples/examples"
    )
    parser.add_argument(
        "--rerun", action="store_true", help="Rerun tests that have already been run"
    )
    parser.add_argument("--rerun_fail", action="store_true", help="Rerun failed tests")
    parser.add_argument("--get_score", action="store_true", help="Get scores")

    # RAG related
    parser.add_argument("--rag", action="store_true", help="Enable RAG context")
    parser.add_argument("--rag_topk", type=int, default=4, help="Top k to use for RAG")
    parser.add_argument(
        "--summarize_rag", action="store_true", help="Summarize RAG context"
    )
    parser.add_argument(
        "--rag_filename",
        type=str,
        default="retrieved_chunk_size_512_chunk_overlap_20_topk_4_embed_bge-large-en-v1.5.txt",
        help="RAG retrieved context file name",
    )

    # Verbose instruction config
    parser.add_argument(
        "--verbose_instruction",
        action="store_true",
        help="Enable verbose instruction loading",
    )

    # Output/logging config
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results/dual_agent",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments to run in parallel",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    # lm config
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--model_provider", type=str, default="openai")
    parser.add_argument(
        "--model_url",
        type=str,
        default="",
        help="The URL of the main generation model API.",
    )
    parser.add_argument(
        "--model_api_key",
        type=str,
        default="",
        help="The API key of the main generation model.",
    )
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=1,
        help="Temperature to fix the generation model at (e.g. o3 can only be run with 1.0)",
    )

    # grounding model config
    parser.add_argument(
        "--ground_provider",
        type=str,
        help="The provider for the grounding model",
    )
    parser.add_argument("--ground_url", type=str, help="The URL of the grounding model")
    parser.add_argument(
        "--ground_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )
    parser.add_argument(
        "--ground_model",
        type=str,
        help="The model name for the grounding model",
    )
    parser.add_argument(
        "--grounding_width",
        type=int,
        default=1280,
        help="Width of screenshot image after processor rescaling",
    )
    parser.add_argument(
        "--grounding_height",
        type=int,
        default=720,
        help="Height of screenshot image after processor rescaling",
    )

    #agent s2 config
    parser.add_argument("--endpoint_provider", type=str, default="")
    parser.add_argument("--endpoint_url", type=str, default="")
    parser.add_argument(
        "--endpoint_api_key",
        type=str,
        default="",
        help="The API key of the grounding model.",
    )
    parser.add_argument("--kb_name", default="kb_s2", type=str, help="Knowledge base name for Agent S2")

    parser.add_argument("--use_schema_induction", action="store_true", help="Use schema induction")

    parser.add_argument("--crop_roi", action="store_true",
                       help="Crop images to change ROI before evaluation (reduces token usage)")
    parser.add_argument("--roi_margin", type=int, default=50,
                       help="Margin around ROI when cropping (default: 50)")
    parser.add_argument("--consolidate_period", type=int, default=5,
                       help="Period to consolidate (default: 5)")
    parser.add_argument("--bash_timeout", type=int, default=60,
                       help="Timeout for bash script execution in seconds (default: 300)")
    parser.add_argument("--wo_episodic", action="store_true",
                       help="Skip episodic encoding and use full conversation history")
    parser.add_argument("--wo_consolidation", action="store_true",
                       help="Disable periodic consolidation and use sliding window")
    parser.add_argument("--sliding_window_size", type=int, default=5,
                       help="Sliding window size (number of conversation turns to keep) (default: 5)")
    parser.add_argument("--schema_dir", type=str, default="D:/projects/qdrant/qdrant_storage", help="Qdrant storage directory")
    parser.add_argument("--use_qdrant_server", action="store_true", help="Use Qdrant server, otherwise use local file storage")
    parser.add_argument("--qdrant_server_url", type=str, default="http://localhost:6333", help="Qdrant server URL")

    args = parser.parse_args()

    if not os.path.exists(args.path_to_vm):
        args.path_to_vm = "./vmware_vm_data/Ubuntu0/Ubuntu0.vmx"

    # Setup logger using utils.setup_logger
    global logger
    result_name = os.path.basename(args.result_dir)
    logger = setup_logger(result_name, args.log_level)

    # Load test metadata for multi-env mode
    if args.num_envs > 1 and not args.get_score:
        # Register signal handlers for graceful termination
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Load test metadata
        with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)
        
        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}
        
        # Save args to json
        path_to_args = os.path.join(args.result_dir, "args.json")
        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        with open(path_to_args, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4)
        
        # Filter tasks using unified logic
        tasks_to_run = filter_tasks(args, test_all_meta, logger)
        
        if not tasks_to_run:
            logger.info("No tasks to process. All tasks have already been completed.")
            sys.exit(0)
        
        # Log tasks info
        task_count_by_domain = {}
        for domain, example_id in tasks_to_run:
            if domain not in task_count_by_domain:
                task_count_by_domain[domain] = 0
            task_count_by_domain[domain] += 1
        
        left_info = ""
        for domain, count in task_count_by_domain.items():
            left_info += f"{domain}: {count}\n"
        logger.info(f"Tasks to process:\n{left_info}")
        
        # Run with multiple environments
        try:
            run_multienv(args, tasks_to_run)
        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt.")
        except Exception as e:
            logger.error(f"Unexpected error in main process: {e}", exc_info=True)
            signal_handler(signal.SIGTERM, None)
        finally:
            # Final cleanup
            logger.info("Main process final cleanup...")
            for env in active_environments:
                if env is not None:
                    try:
                        logger.info(f"Closing environment in final cleanup...")
                        try:
                            env.close()
                            logger.info(f"Environment closed successfully in final cleanup")
                        except Exception as close_error:
                            error_msg = str(close_error)
                            if "not powered on" in error_msg or "not running" in error_msg:
                                logger.info("VM already stopped, skipping close")
                            else:
                                logger.warning(f"Error closing environment: {close_error}")
                        
                        # Clean VMware lock files
                        vm_dir = os.path.dirname(args.path_to_vm)
                        try:
                            env.clean_lock(vm_dir)
                            logger.info("Lock files cleaned in final cleanup")
                        except Exception as lock_error:
                            logger.debug(f"Error cleaning locks: {lock_error}")
                    except Exception as e:
                        logger.error(f"Unexpected error during final cleanup: {e}")
            
            # Terminate processes
            for p in processes:
                if p is not None and p.is_alive():
                    try:
                        logger.info(f"Terminating process {p.name}...")
                        p.terminate()
                    except Exception as e:
                        logger.error(f"Error terminating process: {e}")
            
            time.sleep(1)
            
            # Force kill if needed
            for p in processes:
                if p is not None and p.is_alive():
                    try:
                        logger.info(f"Force killing process {p.name}...")
                        p.kill()
                    except Exception as e:
                        logger.error(f"Error force killing process: {e}")
    else:
        # Single environment mode - use unified task filtering
        # Load test metadata
        with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)
        
        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}
        
        # If only getting scores, skip execution
        if args.get_score:
            
            summary(args.result_dir, test_all_meta)
            sys.exit(0)
        
        # Filter tasks using unified logic
        tasks_to_run = filter_tasks(args, test_all_meta, logger)
        
        if not tasks_to_run:
            logger.info("No tasks to process. All tasks have already been completed.")
            summary(args.result_dir, test_all_meta)
            sys.exit(0)
        
        # Log tasks info
        task_count_by_domain = {}
        for domain, example_id in tasks_to_run:
            if domain not in task_count_by_domain:
                task_count_by_domain[domain] = 0
            task_count_by_domain[domain] += 1
        
        left_info = ""
        for domain, count in task_count_by_domain.items():
            left_info += f"{domain}: {count}\n"
        logger.info(f"Tasks to process:\n{left_info}")
        
        # Create environment
        try:
            from desktop_env.envs.desktop_env import DesktopEnv
        except:
            from desktop_env.desktop_env import DesktopEnv
        try:
            env = DesktopEnv(
                provider_name=args.provider_name,
                path_to_vm=args.path_to_vm,
                action_space=args.action_space,
                snapshot_name=args.snapshot_name,
                headless=args.headless,
                require_a11y_tree=False,
                enable_proxy=False,
                screen_size=(args.screen_width, args.screen_height)
            )
        except:
            env = DesktopEnv(
                path_to_vm=args.path_to_vm,
                snapshot_name=args.snapshot_name,
                action_space=args.action_space,
                headless=args.headless,
                require_a11y_tree=False,
                screen_size=(args.screen_width, args.screen_height)
            )
        args.env = env

        # Import run function
        if args.method == "spider2v_agent":
            from run_spider2v_agent import run
        elif args.method == "coact":
            from run_coact import run
        elif args.method == "agents3":
            from run_agents3 import run
        elif args.method == "agents2":
            from run_agents2 import run
        elif args.method == "cmm":
            from run_cmm import run
        elif args.method == "gta1":
            from run_gta1_agent import run
        else:
            raise ValueError(f"Invalid method: {args.method}")
        
        # Execute tasks one by one
        from tqdm import tqdm
        for domain, example_id in tqdm(tasks_to_run, desc="Processing tasks"):
            logger.info(f"Processing {domain}/{example_id}")
            try:
                # Create tasks list with only this task
                single_task = [(domain, example_id)]
                run(args, logger=logger, tasks=single_task)
            except Exception as e:
                logger.error(f"Error processing {domain}/{example_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Save error information
                example_result_dir = os.path.join(args.result_dir, domain, example_id)
                os.makedirs(example_result_dir, exist_ok=True)
                with open(os.path.join(example_result_dir, "result.txt"), "w") as f:
                    f.write("0.0")
                with open(os.path.join(example_result_dir, "err_reason.txt"), "w") as f:
                    f.write(f"Fatal error: {str(e)}\n\n{traceback.format_exc()}")
        
        # Cleanup
        try:
            try:
                env.close()
            except Exception as close_error:
                error_msg = str(close_error)
                if "not powered on" in error_msg or "not running" in error_msg:
                    logger.info("VM already stopped, skipping close")
                else:
                    logger.warning(f"Error closing environment: {close_error}")
            
            # Clean VMware lock files
            vm_dir = os.path.dirname(args.path_to_vm)
            try:
                env.clean_lock(vm_dir)
                logger.info("Lock files cleaned")
            except Exception as lock_error:
                logger.debug(f"Error cleaning locks: {lock_error}")
        except Exception as e:
            logger.error(f"Unexpected error during cleanup: {e}")
        
        # Show summary
        summary(args.result_dir, test_all_meta)

if __name__ == "__main__":
    run()
