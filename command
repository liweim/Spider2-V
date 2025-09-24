python run_spider2v_agent.py --snapshot init_state --model gpt-4o-2024-05-13 --headless --action_space pyautogui --observation_space som --execution_feedback --rag --result_dir ./results --example evaluation_examples/test_small.json

python run_coact.py --result_dir results/coact_15_10_10_20 --snapshot add_bash --orchestrator_max_steps 15 --coding_max_steps 10 --cua_max_steps 10 --cut_off_steps 20 --test_all_meta_path evaluation_examples/test_small.json

python run_coact.py --result_dir results/coact_15_10_10_20_rag --snapshot add_bash --orchestrator_max_steps 15 --coding_max_steps 10 --cua_max_steps 10 --cut_off_steps 20 --rag --test_all_meta_path evaluation_examples/test_one.json

#复现过程
打开虚拟机改里面的内容，更新代码：/home/user/server/main.py，保存快照，确保分辨率为1920x1080
sudo systemctl daemon-reload
sudo systemctl enable osworld_server@:0.service #/etc/systemd/system/osworld_server@.service
sudo systemctl start osworld_server@:0.service