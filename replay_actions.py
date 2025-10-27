import json
import glob
from desktop_env.envs.desktop_env import DesktopEnv
import os
import time
import tqdm
import sys
sys.path.append('../GUIAgent')
from utils import summary

def replay_actions():
    result_dir = 'results/langgraph_gpt5_cua_50_summarize_rag_verbose'

    env = DesktopEnv(
            path_to_vm="./vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx",
            snapshot_name="config",
            action_space="pyautogui",
            headless=True,
            require_a11y_tree=False,
        )

    for path in tqdm.tqdm(glob.glob(f'{result_dir}/servicenow/*/execution_log.json')):
        data = json.load(open(path, 'r', encoding='utf-8'))
        current_score = data['statistics']['score']
        if current_score > 0:
            continue
        
        print('='*100)
        task_id = os.path.basename(os.path.dirname(path))
        domain = 'servicenow'
        with open(f'evaluation_examples/examples/{domain}/{task_id}/{task_id}.json', 'r', encoding='utf-8', errors='ignore') as inf:
            example = json.load(inf)
        print(f'\x1b[32m[Task instruction for {example["snapshot"]}/{example["id"]}]:\x1b[0m\n\x1b[32m{example["instruction"]}\x1b[0m')
        env.reset(task_config=example)
        action_logs = data['action_logs']
        all_actions = []
        for n, action_log in enumerate(action_logs):
            if 'command' in action_log or 'action' in action_log:
                actions = action_log['command'] if 'command' in action_log else action_log['action']
                if type(actions) != list:
                    actions = [actions]
                all_actions += actions
                for action in actions:
                    print(f'step {n}: {action}')
                    if action.lower() not in ['fail', 'none', 'done']:
                        obs, reward, done, info = env.step(action, 3)
        if len(all_actions) == 0:
            print(f'no actions found for {task_id}')
            continue

        time.sleep(30)
        score = env.evaluate()
        data['statistics']['score'] = score
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        result_path = os.path.join(result_dir, domain, task_id, 'result.txt')
        with open(result_path, 'w') as f:
            f.write(f'{score}')
        print(f'score: {score}')
    env.close()

    with open('evaluation_examples/test_small.json', "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)
    summary(result_dir, test_all_meta)

if __name__ == '__main__':
    replay_actions()