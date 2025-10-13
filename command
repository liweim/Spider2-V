python run_spider2v_agent.py --result_dir results/som_gpt_4o_rag_ef_15_verbose --snapshot config --model gpt-4o --action_space pyautogui --observation_space som --max_steps 15 --execution_feedback --rag --verbose_instruction --example evaluation_examples/test_small.json --headless

python run_my_agent.py --result_dir results/my_agent_o4mini_50_rag_verbose --snapshot config --coordinator_model o4-mini --max_steps 50 --rag --verbose_instruction --test_all_meta_path evaluation_examples/test_small.json --headless

python run_langgraph_agent.py --result_dir results/langgraph_gpt5_cua_50_summarize_rag_verbose --snapshot config --coordinator_model gpt-5 --operator_model computer-use-preview --max_steps 50 --summarize_rag --rag --verbose_instruction --test_all_meta_path evaluation_examples/test_small.json --headless

python run_all.py --method langgraph_agent --result_dir results/langgraph_o4mini_cua_50_rag_verbose --snapshot config --coordinator_model o4-mini --operator_model computer-use-preview --max_steps 50 --rag --verbose_instruction --test_all_meta_path evaluation_examples/test_small.json --headless --rerun

python run_all.py --method coact --result_dir results/coact_15_20_25_50_rag_verbose --snapshot config --orchestrator_max_steps 15 --coding_max_steps 20 --cua_max_steps 25 --cut_off_steps 50 --test_all_meta_path evaluation_examples/test_small.json --headless --rerun

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

# OSWorld上需要安装dos2unix