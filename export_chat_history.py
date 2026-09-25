"""
Chat History Exporter for BreakingBadV3 Project
Parses full conversation transcript logs and outputs both Markdown and HTML exports.
"""

import os
import json
import re
import html
from datetime import datetime

TRANSCRIPT_PATH = r"C:\Users\GushEx\.gemini\antigravity-cli\brain\bb7d6c48-dde8-453e-8628-0fdb3514d0ff\.system_generated\logs\transcript_full.jsonl"
OUT_MD = r"C:\Users\GushEx\Documents\IQOPTIONS BOT\BreakingBadV3\CHAT_HISTORY_EXPORT.md"
OUT_HTML = r"C:\Users\GushEx\Documents\IQOPTIONS BOT\BreakingBadV3\CHAT_HISTORY_EXPORT.html"

def parse_transcript(file_path):
    turns = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            t_type = obj.get("type")
            source = obj.get("source")
            created_at = obj.get("created_at", "")
            content = obj.get("content", "")

            if t_type == "USER_INPUT" and source == "USER_EXPLICIT":
                if "<USER_REQUEST>" in content:
                    m = re.search(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", content, re.DOTALL)
                    user_text = m.group(1).strip() if m else content.strip()
                else:
                    user_text = content.strip()
                
                # Clean up prompt wrappers if present
                turns.append({
                    "role": "user",
                    "time": created_at,
                    "text": user_text
                })

            elif t_type == "PLANNER_RESPONSE" and source == "MODEL":
                if content and content.strip():
                    if turns and turns[-1]["role"] == "assistant":
                        if content.strip() != turns[-1]["text"]:
                            turns[-1]["text"] += "\n\n" + content.strip()
                    else:
                        turns.append({
                            "role": "assistant",
                            "time": created_at,
                            "text": content.strip()
                        })
    return turns

def generate_markdown(turns, out_path):
    lines = [
        "# 📜 BreakingBad V3 — Full Project Chat & Development History",
        "",
        f"> **Generated On**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> **Total Messages**: {len(turns)} ({len([t for t in turns if t['role'] == 'user'])} User Prompts, {len([t for t in turns if t['role'] == 'assistant'])} Assistant Responses)",
        f"> **Project**: BreakingBad V3 Algorithmic Trading & Telegram Ecosystem",
        "",
        "---",
        "",
        "## 📑 Table of Contents (Chronological Overview)",
        ""
    ]

    # Generate TOC
    user_idx = 0
    for i, t in enumerate(turns):
        if t["role"] == "user":
            user_idx += 1
            t_title = t["text"].split("\n")[0][:80].replace("[", "").replace("]", "").replace("`", "")
            t_time = t["time"][:16].replace("T", " ") if t["time"] else ""
            anchor = f"prompt-{user_idx}"
            lines.append(f"{user_idx}. [{t_time}] [{t_title}](#{anchor})")

    lines.append("")
    lines.append("---")
    lines.append("")

    user_idx = 0
    for t in turns:
        t_time = t["time"][:19].replace("T", " ") if t["time"] else "N/A"
        if t["role"] == "user":
            user_idx += 1
            lines.append(f'<a id="prompt-{user_idx}"></a>')
            lines.append(f"### 👤 User Prompt #{user_idx} — *{t_time}*")
            lines.append("")
            lines.append(t["text"])
            lines.append("")
            lines.append("---")
            lines.append("")
        else:
            lines.append(f"#### 🤖 Assistant (Antigravity) — *{t_time}*")
            lines.append("")
            lines.append(t["text"])
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Markdown export saved to: {out_path}")

def generate_html(turns, out_path):
    user_turns_count = len([t for t in turns if t["role"] == "user"])
    asst_turns_count = len([t for t in turns if t["role"] == "assistant"])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreakingBad V3 — Chat & Engineering History</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-user: #1f2937;
            --bg-assistant: #111827;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --text-heading: #58a6ff;
            --accent-green: #238636;
            --accent-blue: #388bfd;
            --code-bg: #0b0e14;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        header h1 {{
            color: #ffffff;
            margin-bottom: 8px;
            font-size: 24px;
        }}
        header .meta {{
            color: #8b949e;
            font-size: 14px;
        }}
        .chat-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .message-card {{
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .message-card.user {{
            background-color: var(--bg-user);
            border-left: 4px solid var(--accent-blue);
        }}
        .message-card.assistant {{
            background-color: var(--bg-assistant);
            border-left: 4px solid var(--accent-green);
        }}
        .message-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 8px;
        }}
        .sender-badge {{
            font-weight: 600;
            font-size: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .message-card.user .sender-badge {{ color: #79c0ff; }}
        .message-card.assistant .sender-badge {{ color: #7ee787; }}
        .timestamp {{
            font-size: 12px;
            color: #8b949e;
        }}
        .message-body {{
            font-size: 14.5px;
            overflow-x: auto;
        }}
        .message-body p {{
            margin-bottom: 12px;
        }}
        .message-body pre {{
            background-color: var(--code-bg);
            padding: 14px;
            border-radius: 6px;
            border: 1px solid #21262d;
            overflow-x: auto;
            margin: 14px 0;
        }}
        .message-body code {{
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 13px;
        }}
        .message-body table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        .message-body th, .message-body td {{
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            text-align: left;
        }}
        .message-body th {{
            background-color: #1f242c;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📜 BreakingBad V3 — Project Chat & Engineering Transcript</h1>
            <div class="meta">
                <span><b>Export Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span> • 
                <span><b>User Prompts:</b> {user_turns_count}</span> • 
                <span><b>Assistant Turns:</b> {asst_turns_count}</span>
            </div>
        </header>

        <div class="chat-container" id="chat-stream"></div>
    </div>

    <script>
        const chatData = {json.dumps(turns, ensure_ascii=False)};
        const container = document.getElementById('chat-stream');

        chatData.forEach((turn, idx) => {{
            const card = document.createElement('div');
            card.className = `message-card ${{turn.role}}`;

            const header = document.createElement('div');
            header.className = 'message-header';

            const sender = document.createElement('div');
            sender.className = 'sender-badge';
            sender.innerHTML = turn.role === 'user' ? `👤 User #${{idx + 1}}` : '🤖 Antigravity Assistant';

            const time = document.createElement('div');
            time.className = 'timestamp';
            time.textContent = turn.time ? turn.time.replace('T', ' ').substring(0, 19) : '';

            header.appendChild(sender);
            header.appendChild(time);

            const body = document.createElement('div');
            body.className = 'message-body';
            body.innerHTML = marked.parse(turn.text);

            card.appendChild(header);
            card.appendChild(body);
            container.appendChild(card);
        }});
    </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] HTML export saved to: {out_path}")

if __name__ == "__main__":
    turns = parse_transcript(TRANSCRIPT_PATH)
    generate_markdown(turns, OUT_MD)
    generate_html(turns, OUT_HTML)
