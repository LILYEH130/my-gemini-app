import streamlit as st
from google import genai
from google.genai import types

# 1. 網頁基本設定
st.set_page_config(page_title="極省 Token Gemini 助手", layout="centered")
st.title("🚀 極省 Token Gemini 助手")

# 2. 側邊欄設定
with st.sidebar:
    st.header("⚙️ 設定中心")
    api_key = st.text_input("輸入 Google AI Studio API Key (AIzaSy 開頭)", type="password")

    st.markdown("---")
    st.subheader("🪙 Token 節省設定")
    max_history = st.slider("保留對話輪數 (滑動視窗)", min_value=1, max_value=10, value=3)
    max_tokens = st.slider("單次回答最大 Token 限制", min_value=100, max_value=2000, value=800)  # 建議拉大到 800

    # 核心除錯工具：手動清除歷史紀錄，防止舊錯誤干擾
    st.markdown("---")
    if st.button("🗑️ 清除所有對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# 3. 初始化對話紀錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. 顯示歷史對話
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 5. 處理使用者輸入
if user_input := st.chat_input("請輸入您的問題..."):
    if not api_key:
        st.error("請先在左側邊欄輸入您的免費 Gemini API Key！")
        st.stop()

    # 顯示使用者發言并存入紀錄
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ------ 核心省 Token 機制（精準對齊官方格式，防止中英混雜） ------
    formatted_contents = []
    # 嚴格擷取最近 N 輪對話
    recent_history = st.session_state.messages[-(max_history * 2):]

    for msg in recent_history:
        # 確保角色代號只有 user 或 model
        api_role = "user" if msg["role"] == "user" else "model"
        formatted_contents.append(
            types.Content(
                role=api_role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )
    # ------------------------------------------------------------------

    # 6. 呼叫 AI API (強制繁體中文與穩定輸出)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            client = genai.Client(api_key=api_key)

            # 使用最新的 3.6-flash 穩定通道
            response = client.models.generate_content_stream(
                model='gemini-3.6-flash',
                contents=formatted_contents,
                config=types.GenerateContentConfig(
                    # 嚴格命令 AI 必須使用繁體中文，且不要中斷
                    system_instruction=(
                        "你是一位專業、溫柔且富有同理心的助手。面對法律程序或具備情緒壓力的議題時，"
                        "請使用結構清晰、條列分明且有溫度的『繁體中文（台灣）』進行完整回答。"
                        "嚴禁回答任何英文片段或中斷語句。"
                    ),
                    max_output_tokens=max_tokens,
                    temperature=0.3,  # 降低隨機性，讓回答更穩定嚴謹
                ),
            )

            for chunk in response:
                if chunk and hasattr(chunk, 'text') and chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")

            # 確保有字數才儲存
            if full_response.strip():
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                message_placeholder.error("伺服器回應為空，請再試一次。")

        except Exception as e:
            st.error(f"呼叫失敗。錯誤訊息: {str(e)}")
