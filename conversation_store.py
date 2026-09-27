from collections import defaultdict

conversations = defaultdict(list)

def add_message(session_id, role, content):
    conversations[session_id].append({
        "role": role,
        "content": content,
    })

def get_history(session_id):
    return conversations[session_id]

def clear_history(session_id):
    conversations.pop(session_id, None)
