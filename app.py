import time
import random

import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("智扫通机器人智能客服")
st.divider()

# 初始化 session 状态
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "user_id" not in st.session_state:
    st.session_state["user_id"] = f"user_{random.randint(1000, 9999)}"

if "history" not in st.session_state:
    st.session_state["history"] = []

if "message" not in st.session_state:
    st.session_state["message"] = []

# 侧边栏：结束对话按钮
with st.sidebar:
    st.write(f"用户ID：{st.session_state['user_id']}")
    if st.button("结束对话"):
        agent = st.session_state["agent"]
        agent.memory_manager.on_session_end(
            st.session_state["user_id"], st.session_state["history"]
        )
        st.session_state["history"] = []
        st.session_state["message"] = []
        st.rerun()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(
            prompt,
            user_id=st.session_state["user_id"],
            history=st.session_state["history"],
        )

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()