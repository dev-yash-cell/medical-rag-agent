import streamlit as st
import requests

# UI Configuration
st.set_page_config(page_title="Clinical Operations RAG", page_icon="🏥")
st.title("🏥 Medical Operations Support Bot")
st.markdown("Ask questions about hospital compliance, oncology guidelines, or billing codes.")

# Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("E.g., What is the protocol for equipment failure?"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Communicate with FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Searching compliance manuals..."):
            try:
                response = requests.post(
                    "http://localhost:8000/ask", 
                    json={"question": prompt},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer")
                    citation = data.get("citation")
                    
                    # Formatting the output to highlight the strict citation requirement
                    full_response = f"{answer}\n\n**Source:** `{citation}`"
                    st.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Backend Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Is FastAPI running on port 8000?")