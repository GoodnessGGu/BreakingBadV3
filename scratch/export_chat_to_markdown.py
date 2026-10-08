import json
import os
import re

def export_chat():
    full_path = r"C:\Users\GushEx\.gemini\antigravity-cli\brain\bb7d6c48-dde8-453e-8628-0fdb3514d0ff\.system_generated\logs\transcript_full.jsonl"
    short_path = r"C:\Users\GushEx\.gemini\antigravity-cli\brain\bb7d6c48-dde8-453e-8628-0fdb3514d0ff\.system_generated\logs\transcript.jsonl"
    
    src = full_path if os.path.exists(full_path) else short_path
    output_dir = os.path.join(os.getcwd(), "docs")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "CHAT_HISTORY_EXPORT.md")

    turns = []
    current_user_msg = None
    current_asst_msgs = []

    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            
            step_type = data.get("type")
            content = (data.get("content") or "").strip()
            created_at = data.get("created_at", "")
            
            # Skip system internal instructions or tool status messages in USER_INPUT
            if step_type == "USER_INPUT":
                if not content:
                    continue
                # If there was an existing user message, save it along with its assistant response
                if current_user_msg:
                    turns.append({
                        "user": current_user_msg,
                        "assistant": "\n\n".join(current_asst_msgs).strip()
                    })
                current_user_msg = {"text": content, "time": created_at}
                current_asst_msgs = []
            elif step_type == "PLANNER_RESPONSE":
                # Only include assistant text if it's not empty and not just raw tool execution metadata
                if content and not data.get("tool_calls"):
                    current_asst_msgs.append(content)

    if current_user_msg:
        turns.append({
            "user": current_user_msg,
            "assistant": "\n\n".join(current_asst_msgs).strip()
        })

    print(f"Extracted {len(turns)} total conversation turns.")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# BreakingBad V3 - Complete Chat & Development History Export\n\n")
        f.write(f"> **Export Date:** {created_at or '2026-10-08'}\n")
        f.write(f"> **Total Turns:** {len(turns)}\n")
        f.write(f"> **Conversation ID:** `bb7d6c48-dde8-453e-8628-0fdb3514d0ff`\n\n")
        f.write("---\n\n")

        for idx, turn in enumerate(turns, 1):
            u_text = turn["user"]["text"]
            u_time = turn["user"]["time"]
            a_text = turn["assistant"]

            f.write(f"## Turn {idx} ({u_time})\n\n")
            f.write(f"### 👤 User\n\n{u_text}\n\n")
            if a_text:
                f.write(f"### 🤖 Assistant\n\n{a_text}\n\n")
            else:
                f.write(f"### 🤖 Assistant\n\n*(Tool execution / code updates performed)*\n\n")
            f.write("---\n\n")

    print(f"Export successfully saved to: {out_file}")
    print(f"File size: {os.path.getsize(out_file) / 1024:.1f} KB")

if __name__ == "__main__":
    export_chat()
