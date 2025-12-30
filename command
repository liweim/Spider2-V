python run_spider2v_agent.py --result_dir results/som_gpt_4o_rag_ef_15_verbose --snapshot config --model gpt-4o --action_space pyautogui --observation_space som --max_steps 15 --execution_feedback --rag --verbose_instruction --example evaluation_examples/test_small.json --headless

python run_my_agent.py --result_dir results/my_agent_o4mini_50_rag_verbose --snapshot config --coordinator_model o4-mini --max_steps 50 --rag --verbose_instruction --test_all_meta_path evaluation_examples/test_small.json --headless

python run_langgraph_agent.py --result_dir results/langgraph_gpt5_cua_50_summarize_rag_verbose --snapshot config --coordinator_model gpt-5 --operator_model computer-use-preview --max_steps 50 --summarize_rag --rag --verbose_instruction --test_all_meta_path evaluation_examples/test_small.json --headless

#复现过程
打开虚拟机改里面的内容，更新代码：/home/user/server/main.py，保存快照，确保分辨率为1920x1080
sudo systemctl restart osworld_server@:0.service

## servicenow
pip install browsergym-workarena==0.2.1 #版本改了就验证对不上了
playwright install
$env:SNOW_INSTANCE_URL="https://empmassimo12.service-now.com/"
$env:SNOW_INSTANCE_UNAME="admin"
$env:SNOW_INSTANCE_PWD="^lq3Q+XKK9?n"
workarena-install

## dbt
需要双重验证且pydantic<2，与autogen冲突

## snowflake
只有一个月试用

## bigquery
需要双重验证

# OSWorld上需要安装dos2unix