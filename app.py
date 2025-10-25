import streamlit as st
from gpgchat_tou import (
    load_memory, save_memory,
    estimate_usage_from_bill, retrieve_utility_info,
    estimate_new_bill, generate_combined_gpt_tips, interactive_gpt
)
import time, os

st.set_page_config(page_title="Energy Efficiency Assistant", page_icon="⚡")
st.title("⚡ GPG-Chat: Your Energy Efficiency Assistant")

# --- Memory setup ---
memory = load_memory()
chat_memory_file = "chat_memory.txt"
if not os.path.exists(chat_memory_file):
    open(chat_memory_file, "w").close()  # create empty chat file if not exists

# --- Sidebar ---
with st.sidebar:
    st.header("🧠 User Info Memory")
    st.json(memory)

    # --- Chat History Display ---
    with st.expander("💬 Chat Memory", expanded=True):
        try:
            chat_content = open(chat_memory_file, "r").read().strip()
            if chat_content:
                st.text_area("Conversation History", chat_content, height=400)
            else:
                st.info("No chat history yet.")
        except Exception as e:
            st.error(f"Error loading chat memory: {e}")

# --- Main Input Form ---
with st.form("user_inputs"):
    user_name = st.text_input("Enter you name", memory.get("user_name", ""))
    zip_code = st.text_input("ZIP Code", memory.get("zip_code", ""))
    old_bill = st.number_input(
        "Previous Bill ($)",
        value=float(memory.get("old_bill", 0)) if memory.get("old_bill") else 0.0
    )
    target_bill = st.number_input(
        "Target Bill ($)",
        value=float(memory.get("target_bill", 0)) if memory.get("target_bill") else 0.0
    )
    utility_name = st.text_input("Utility Provider", memory.get("utility_name", ""))
    user_info = st.text_area("Usage Details (EV, HVAC, etc.)", memory.get("user_info", ""))

    submitted = st.form_submit_button("Generate Tips")

if submitted:
    start_time = time.time()

    # Save memory
    memory.update({
        "user_name": user_name,
        "zip_code": zip_code,
        "old_bill": old_bill,
        "target_bill": target_bill,
        "utility_name": utility_name,
        "user_info": user_info
    })
    save_memory(memory)

    # Compute estimates
    usage_kwh = estimate_usage_from_bill(old_bill)
    query = f"{utility_name} {zip_code} residential rate"
    retrieved_rows = retrieve_utility_info(query, top_k=3)

    new_bill, rate, fixed = estimate_new_bill(usage_kwh, retrieved_rows)
    if new_bill is None:
        st.error("Could not estimate new bill.")
    else:
        reduction_needed = (1 - (float(target_bill) / float(new_bill))) * 100
        tips = generate_combined_gpt_tips(user_info, old_bill, new_bill, usage_kwh)

        #st.success(f"💰 Estimated new bill: **${new_bill:.2f}** at rate ${rate:.4f}/kWh")
        st.write(f"To reach ${target_bill}, reduce or shift **{reduction_needed:.1f}%** of usage.")
        st.subheader(f"💡 Personalized Energy Tips for {user_name}")
        st.write(tips)

        elapsed = time.time() - start_time
        st.caption(f"Response generated in {elapsed:.2f}s")

        # Append to chat memory
        with open(chat_memory_file, "a") as f:
            f.write(f"Assistant (Tips): {tips}\n\n")

# --- Chat Interface ---
st.divider()
st.header("💬 Continue the Conversation")

user_input = st.text_input("You:", "")
if st.button("Send"):
    if user_input.lower() in ["exit", "quit"]:
        st.stop()

    # Load chat context
    context = open(chat_memory_file, "r").read() if os.path.exists(chat_memory_file) else ""
    response = interactive_gpt(user_input, context)

    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)

    # Save chat turn
    with open(chat_memory_file, "a") as f:
        f.write(f"You: {user_input}\nAssistant: {response}\n\n")

