import json
import glob
from desktop_env.envs.desktop_env import DesktopEnv
import os
import time
import re
import tqdm
import sys
sys.path.append('../GUIAgent')
from utils import summary

def replay_actions():
    result_dir = 'results/tool_agent_gpt5_gta1_50'

    env = DesktopEnv(
            path_to_vm="./vm_data/Ubuntu0/Ubuntu0/Ubuntu0.vmx",
            snapshot_name="low_res",
            action_space="pyautogui",
            headless=False,
            require_a11y_tree=False,
            screen_size=(1280, 720)
        )

    for path in tqdm.tqdm(glob.glob(f'{result_dir}/servicenow/5af077c3-24d4-4708-a58b-970ea9c2f111/execution_log.json')):
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
        for n, action_log in enumerate(action_logs):
            if 'action_summary' in action_log:
                action_summary = action_log['action_summary']
                rfind = re.findall(r' \| Code: (.*?) \| Result: ', action_summary, re.DOTALL)
                if len(rfind) == 0:
                    continue
                action = rfind[0]
                print(f'\nstep {n+1}: {action}')
                if action.lower() not in ['fail', 'none', 'done']:
                    obs, reward, done, info = env.step(action, 3)

        score = env.evaluate()
        data['statistics']['score'] = score
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        result_path = os.path.join(result_dir, domain, task_id, 'result.txt')
        with open(result_path, 'w') as f:
            f.write(f'{score}')
        print(f'score: {score}')
    env.close()

    with open('evaluation_examples/test_abstract.json', "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)
    summary(result_dir, test_all_meta)

if __name__ == '__main__':
    replay_actions()