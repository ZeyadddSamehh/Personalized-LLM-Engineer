import streamlit as st
import time

# ==========================================
# SESSION STATE SETUP (The "Filing Cabinet")
# ==========================================
# 1. Create a filing cabinet to hold multiple conversations.
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {"Chat 1": []} # Starts with one empty folder

# 2. Keep track of which folder we are currently looking at
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"


# ==========================================
# SIDEBAR SECTION
# ==========================================
with st.sidebar:
    st.title("History")
    
    # 1. NEW CHAT BUTTON
    if st.button("➕ New Chat", use_container_width=True):
        # Figure out what number to name the next chat (e.g., Chat 2, Chat 3)
        new_chat_num = len(st.session_state.all_chats) + 1
        new_chat_name = f"Chat {new_chat_num}"
        
        # Create a new empty list in our filing cabinet for this chat
        st.session_state.all_chats[new_chat_name] = []
        
        # Switch our view to this new chat
        st.session_state.current_chat = new_chat_name
        st.rerun() # Refresh the page instantly
    
    st.divider() # Draw a horizontal line
    
    # 2. CHAT HISTORY LIST
    st.write("### Past Chats")
    
    # Loop through every chat name in our filing cabinet
    for chat_name in st.session_state.all_chats.keys():
        # Create a button for each past chat
        if st.button(chat_name, use_container_width=True):
            # If the user clicks it, switch the view to that chat
            st.session_state.current_chat = chat_name
            st.rerun()

# ==========================================
# MAIN CHAT SECTION
# ==========================================
# The title changes based on which chat you are in!
st.title(f"🤖 {st.session_state.current_chat}")

# Go into the filing cabinet and grab the messages for whichever chat we are looking at
current_messages = st.session_state.all_chats[st.session_state.current_chat]

# Draw those messages on the screen
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The Chat Input Box
if prompt := st.chat_input("Type your message here..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save the user's message to the correct folder in the filing cabinet
    st.session_state.all_chats[st.session_state.current_chat].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # --- FAKE AI RESPONSE ---
        time.sleep(1) 
        fake_reply = f"I am a fake AI in {st.session_state.current_chat}. You just said: '{prompt}'"
        # ------------------------

        message_placeholder.markdown(fake_reply)

    # Save the AI's message to the correct folder
    st.session_state.all_chats[st.session_state.current_chat].append({"role": "assistant", "content": fake_reply})