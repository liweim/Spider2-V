from desktop_env.envs.desktop_env import DesktopEnv
import json
import random

def setup():
    # feel free to change the example!
    # task instruction: Help me materialize the asset top10_story_ids in this dagster project in the UI. Do NOT materialize other assets.
    example_path = 'evaluation_examples/examples/dagster/22ef9058-6188-422a-9c12-e6934e4ed936/22ef9058-6188-422a-9c12-e6934e4ed936.json'
    with open(example_path, 'r') as infile:
        example = json.load(infile)

    env = DesktopEnv(action_space="pyautogui")

    obs = env.reset(task_config=example)
    print(f'Task instruction: {example["instruction"]}')
    obs, reward, done, info = env.step("pyautogui.rightClick()")
    input('Now, you can finish the task in the virtual machine manually and Press ENTER to evaluate ...')
    score = env.evaluate()
    print(f'Evaluation score: {float(score):.1f}')
    env.close()

def sample():
    # data = json.load(open("evaluation_examples/test_non_account.json"))
    # sample_data = {}
    # count = 0
    # for domain, items in data.items():
    #     if len(items) > 1:
    #         sample_data[domain] = [items[0]]
    #         count += 1
    #         for item in items[1:]:
    #             if random.random() < 0.1:   
    #                 count += 1
    #                 sample_data[domain].append(item)
    # print(f"Sampled {count} items")
    # json.dump(sample_data, open("evaluation_examples/test_small.json", "w"), indent=4)

    data = json.load(open("evaluation_examples/test_account.json"))
    sample_data = json.load(open("evaluation_examples/test_small.json"))
    count = 0
    for domain, items in data.items():
        if len(items) > 1:
            if domain not in sample_data:
                sample_data[domain] = [items[0]]
                count += 1
            for item in items[1:]:
                if random.random() < 0.1:   
                    count += 1
                    sample_data[domain].append(item)
    print(f"Sampled {count} items")
    json.dump(sample_data, open("evaluation_examples/test_small.json", "w"), indent=4)

    # data = json.load(open("evaluation_examples/test_account.json"))
    # sample_data = {}
    # count = 0
    # for domain, items in data.items():
    #     if len(items) > 1:
    #         sample_data[domain] = [items[0]]
    #         count += 1
    # print(f"Sampled {count} items")
    # json.dump(sample_data, open("evaluation_examples/check_account.json", "w"), indent=4)

if __name__ == "__main__":
    # setup()
    sample()