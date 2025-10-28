import json
from json_repair import repair_json
import glob
import shutil
import re
import os
from desktop_env.envs.desktop_env import DesktopEnv
import time
import pandas as pd

def tmp1():
    for path in glob.glob('results/som_gpt_4o_rag_ef_15_verbose/*/*/execution_log.json'):
        data = json.load(open(path, 'r', encoding='utf-8'))
        # steps = data['statistics']['total_steps']
        # if steps == 0:
        #     image_count = 0
        # elif steps == 1:
        #     image_count = 1
        # elif steps == 2:
        #     image_count = 1+2
        # elif steps == 3:
        #     image_count = 1+2+3
        # else:
        #     image_count = (steps-3)*4+6
        # data['statistics']['image_count'] = image_count
        data['statistics']['total_cost'] = (data['statistics']['prompt_tokens'] * 2.5 + data['statistics']['completion_tokens'] * 10)/1000000+data['statistics']['image_count']*3.613/1000
        json.dump(data, open(path, 'w', encoding='utf-8'), indent=2)

def tmp2():
    for path in glob.glob('results/coact_15_20_25_50_rag_verbose/*/*/execution_log.json'):
        data = json.load(open(path, 'r', encoding='utf-8'))
        data['statistics']['prompt_tokens'] = data['statistics']['prompt_tokens']*8
        data['statistics']['completion_tokens'] = data['statistics']['completion_tokens']*8
        data['statistics']['total_cost'] = (data['statistics']['prompt_tokens'] * 5 + data['statistics']['completion_tokens'] * 15)/1000000+data['statistics']['image_count']*7.225/1000
        json.dump(data, open(path, 'w', encoding='utf-8'), indent=2)

def tmp3():
    verbose_data = json.load(open(r'D:\projects\Spider2-V\evaluation_examples\test_verbose.json', 'r', encoding='utf-8'))
    small_data = json.load(open(r'D:\projects\Spider2-V\evaluation_examples\test_small.json', 'r', encoding='utf-8'))
    verbose_small_data = {}
    count = 0
    for domain, lines in small_data.items():
        for line in lines:
            if line in verbose_data[domain]:
                if domain not in verbose_small_data:
                    verbose_small_data[domain] = []
                verbose_small_data[domain].append(line)
                count += 1
    json.dump(verbose_small_data, open(r'D:\projects\Spider2-V\evaluation_examples\test_verbose_small.json', 'w', encoding='utf-8'), indent=2)
    print(count)

def tmp4():
    abstract_data = json.load(open(r'D:\projects\Spider2-V\evaluation_examples\test_abstract.json', 'r', encoding='utf-8'))
    small_data = json.load(open(r'D:\projects\Spider2-V\evaluation_examples\test_small.json', 'r', encoding='utf-8'))
    abstract_small_data = {}
    count = 0
    for domain, lines in small_data.items():
        for line in lines:
            if line in abstract_data[domain]:
                if domain not in abstract_small_data:
                    abstract_small_data[domain] = []
                abstract_small_data[domain].append(line)
                count += 1
    json.dump(abstract_small_data, open(r'D:\projects\Spider2-V\evaluation_examples\test_abstract_small.json', 'w', encoding='utf-8'), indent=2)
    print(count)

def tmp5():
    for path in glob.glob('results/agents3_o4mini_cua_50_summarize_rag/*/*/execution_log.json'):
        origin_path = path.replace('execution_log.json', 'execution_log_original.json')
        with open(origin_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Deep copy to avoid modifying the original data dictionary in memory
        corrected_data = json.loads(json.dumps(data))

        # Extract relevant sections from the data
        action_logs = corrected_data.get('action_logs', [])
        stats = corrected_data.get('statistics', {})
        model_usage = stats.get('model_usage', {})
        grounding_agent_usage = model_usage.get('grounding_agent', {})

        if not action_logs or not grounding_agent_usage:
            print("Could not find 'action_logs' or 'grounding_agent' stats. Aborting.")
            return data

        # 1. Count the number of actions
        total_steps = stats.get('total_steps', 0)
        
        # Count 'other' actions ('DONE', 'FAIL', 'WAIT')
        other_actions = [log for log in action_logs if log.get('action') in ['DONE', 'FAIL', 'WAIT']]
        other_action_count = len(other_actions)

        # The number of GUI actions is the total number of entries in the action_logs
        # as each step corresponds to a screenshot processed by the grounding agent.
        gui_action_count = len(action_logs) - other_action_count
        
        # Calculate coding actions based on the provided formula
        coding_action_count = total_steps - other_action_count - gui_action_count

        print("--- Action Count Statistics ---")
        print(f"Total Steps: {total_steps}")
        print(f"GUI Actions: {gui_action_count}")
        print(f"Other Actions ('DONE', 'WAIT', 'FAIL'): {other_action_count}")
        print(f"Coding Actions (calculated): {coding_action_count}")
        print("-" * 30)

        # 2. Correct the grounding_agent's prompt_tokens
        original_prompt_tokens = grounding_agent_usage.get('prompt_tokens', 0)
        original_completion_tokens = grounding_agent_usage.get('completion_tokens', 0)
        original_image_count = grounding_agent_usage.get('image_count', 1) # Avoid division by zero

        if original_image_count == 0:
            print("Warning: Original image_count for grounding_agent is zero. Cannot calculate correction.")
            corrected_prompt_tokens = 0
        else:
            corrected_prompt_tokens = int(original_prompt_tokens / original_image_count * gui_action_count)
            corrected_completion_tokens = int(original_completion_tokens / original_image_count * gui_action_count)

        print("\n--- Grounding Agent Correction ---")
        print(f"Original Prompt Tokens: {original_prompt_tokens}")
        print(f"Original Completion Tokens: {original_completion_tokens}")
        print(f"Original Image Count: {original_image_count}")
        print(f"Corrected Prompt Tokens: {corrected_prompt_tokens}")
        print(f"Corrected Completion Tokens: {corrected_completion_tokens}")
        print(f"Corrected Image Count: {gui_action_count}")
        print("-" * 30)

        # Update the dictionary with the new value
        corrected_data['statistics']['model_usage']['grounding_agent']['prompt_tokens'] = corrected_prompt_tokens
        corrected_data['statistics']['model_usage']['grounding_agent']['completion_tokens'] = corrected_completion_tokens
        corrected_data['statistics']['model_usage']['grounding_agent']['image_count'] = gui_action_count
        corrected_data['statistics']['model_usage']['grounding_agent']['cost'] = (corrected_prompt_tokens * 3 + corrected_completion_tokens * 12)/1000000

        corrected_data['statistics']['prompt_tokens'] = sum([usage['prompt_tokens'] for usage in corrected_data['statistics']['model_usage'].values()])
        corrected_data['statistics']['completion_tokens'] = sum([usage['completion_tokens'] for usage in corrected_data['statistics']['model_usage'].values()])
        corrected_data['statistics']['total_cost'] = sum([usage['cost'] for usage in corrected_data['statistics']['model_usage'].values()])
        corrected_data['statistics']['cua_steps'] = gui_action_count
        corrected_data['statistics']['coding_steps'] = coding_action_count
        corrected_data['statistics']['image_count'] = gui_action_count+corrected_data['statistics']['model_usage']['generator_agent']['image_count']+corrected_data['statistics']['model_usage']['reflection_agent']['image_count']


        # print(json.dumps(corrected_data, indent=2, ensure_ascii=False))

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(corrected_data, f, indent=2)

def tmp6():
    for filepath in glob.glob('D:/projects/Spider2-V/logs/agents3_o4mini_cua_50_summarize_rag*.log'):
        # Define a temporary file path in the same directory.
        temp_filepath = filepath + '.tmp'
        
        # Regex pattern to find and replace base64 image data.
        pattern = re.compile(
            r"""data:image/png;base64,([A-Za-z0-9+/=]+)"""
        )
        replacement = r"data:image/png;base64,[BASE64_DATA_REMOVED]"
        
        lines_processed = 0
        lines_cleaned = 0
        
        try:
            # Step 1: Read from the source and write to the temporary file line by line.
            with open(filepath, 'r', encoding='utf-8') as infile, \
                open(temp_filepath, 'w', encoding='utf-8') as outfile:
                
                for line in infile:
                    lines_processed += 1
                    cleaned_line, subs_made = pattern.subn(replacement, line)
                    outfile.write(cleaned_line)
                    if subs_made > 0:
                        lines_cleaned += 1

            # Step 2: Replace the original file with the temporary file.
            # shutil.move is an atomic operation on most systems, making it safer
            # than os.remove followed by os.rename.
            shutil.move(temp_filepath, filepath)

            if lines_cleaned > 0:
                print(f"Lines with Base64 data cleaned: {lines_cleaned}")

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            # If an error occurs, clean up the temporary file if it exists.
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)


def tmp7():
    old = 'agents3_o4mini_cua_50_summarize_rag'
    new = 'agents2_o4mini_cua_50_summarize_rag'
    for path in glob.glob(f'results/{old}/*/*'):
        result_path = path+'/result.txt'
        new_result_path = result_path.replace(old, new)
        result = eval(open(result_path, 'r', encoding='utf-8').read())
        
        exe_path = path+'/execution_log.json'
        new_exe_path = exe_path.replace(old, new)
        if result == 0 and not os.path.exists(new_exe_path):
            print(f'{exe_path} -> {new_exe_path}')
            # exit()
            shutil.copy(exe_path, new_exe_path)
            shutil.copy(result_path, new_result_path)

def tmp8():
    result_name = 'langgraph_gpt5_cua_50_summarize_rag'
    result_path = 'D:/projects/GUIAgent/human_result.xlsx'
    df = pd.read_excel(result_path)
    config = json.load(open('evaluation_examples/test_small.json', 'r', encoding='utf-8'))
    res = []
    for domain, lines in config.items():
        for line in lines:
            path = f'results/{result_name}/{domain}/{line}/result.txt'
            if os.path.exists(path):
                res.append(eval(open(path, 'r', encoding='utf-8').read()))
            else:
                res.append(0)
    df[result_name] = res
    df.to_excel(result_path, index=False)


if __name__ == '__main__':
    i = input('really want to run this? (y/n)')
    if i == 'y':
        tmp8()