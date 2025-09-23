python run_spider2v_agent.py --snapshot init_state --model gpt-4o-2024-05-13 --headless --action_space pyautogui --observation_space som --execution_feedback --rag --result_dir ./results --example evaluation_examples/test_small.json

python run_coact.py --provider_name vmware --path_to_vm vmware_vm_data/Ubuntu0/Ubuntu0.vmx --test_all_meta_path evaluation_examples/test_one_simple.json 

#复现过程
1. enable_proxy=False
2. 打开虚拟机改里面的内容，代码在：/home/user/server/main.py，将_append_event给comment掉，保存快照，确保分辨率为1920x1080
sudo systemctl daemon-reload
sudo systemctl enable osworld.service #/etc/systemd/system/osworld.service
sudo systemctl start osworld.service
3. D:\projects\OSWorld\desktop_env\providers\aws\manager.py里将ami_id改成snapshot名字
4. vmware的password为password