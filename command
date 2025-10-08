python run_spider2v_agent.py --result_dir results/som_gpt_4o_rag_ef_15_verbose --snapshot config --model gpt-4o --action_space pyautogui --observation_space som --max_steps 15 --execution_feedback --rag --verbose_instruction --example evaluation_examples/test_small.json --headless

python run_coact.py --result_dir results/coact_15_20_25_50_rag_verbose --snapshot config --orchestrator_max_steps 15 --coding_max_steps 20 --cua_max_steps 25 --cut_off_steps 50 --rag --verbose_instruction --test_all_meta_path evaluation_examples/test_one.json --headless --rerun

python run_coact.py --result_dir results/coact_15_10_10_20_account --snapshot config --orchestrator_max_steps 15 --coding_max_steps 10 --cua_max_steps 10 --cut_off_steps 20 --test_all_meta_path evaluation_examples/check_account.json --rerun

python run_my_agent.py --result_dir results/my_agent_o4mini_50_rag_verbose --snapshot config --coordinator_model o4-mini --max_steps 50 --rag --verbose_instruction --test_all_meta_path evaluation_examples/test_small.json --headless

python run_langgraph_agent.py --result_dir results/langgraph_o4mini_cua_50_rag_verbose --snapshot config --coordinator_model o4-mini --operator_model computer-use-preview --max_steps 50 --rag --verbose_instruction --test_all_meta_path evaluation_examples/test_small.json --headless

python run_langgraph_agent.py --result_dir results_osworld/langgraph_o4mini_cua_50_rag_verbose --path_to_vm D:/projects/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx --snapshot fix_bug --coordinator_model o4-mini --operator_model computer-use-preview --max_steps 50 --test_config_base_dir D:/projects/OSWorld/evaluation_examples/examples --test_all_meta_path D:/projects/OSWorld/evaluation_examples/test_small.json --headless

#复现过程
打开虚拟机改里面的内容，更新代码：/home/user/server/main.py，保存快照，确保分辨率为1920x1080
sudo systemctl daemon-reload
sudo systemctl enable osworld_server@:0.service #/etc/systemd/system/osworld_server@.service
sudo systemctl start osworld_server@:0.service
把需要账号的全登录一遍

## servicenow
pip install browsergym
playwright install
$env:SNOW_INSTANCE_URL="https://empmassimo12.service-now.com/"
$env:SNOW_INSTANCE_UNAME="admin"
$env:SNOW_INSTANCE_PWD="^lq3Q+XKK9?n"
workarena-install

## dbt，需要双重验证且pydantic<2，与autogen冲突

$env:OPENAI_API_KEY="sk-E7gOgfTjf0tREnYXEa1767178b7f43499eBdA49389CdD905"
$env:OPENAI_API_BASE="https://api2.road2all.com/v1"