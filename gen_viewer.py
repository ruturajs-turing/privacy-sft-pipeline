#!/usr/bin/env python3
"""Generate a polished HTML viewer for privacy SFT trajectories."""
import json, html, re, datetime
from pathlib import Path

OUT = Path(__file__).parent / "output"
DEST = Path(__file__).resolve().parent.parent / "privacy-trajectory-viewer.html"

TASKS = [
    ("T-033-02", "347d195e-031d-4ad4-966b-bf85c050e604"),
    ("T-002-12", "6d20f98f-e5bc-4a0d-b7de-0fe8b4594b86"),
    ("T-042-02", "8cbce21b-6aba-4c56-b71a-2d4c9876cb3b"),
]

trajectories = {}
for task_id, sub_id in TASKS:
    with open(OUT / sub_id / "metadata.json") as f:
        meta = json.load(f)
    events = []
    with open(OUT / sub_id / "trajectory.jsonl") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    trajectories[task_id] = {"meta": meta, "events": events}


def extract_text(content):
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content) if content else ""


def fmt_json(obj, max_len=300):
    s = json.dumps(obj, indent=2, default=str)
    if len(s) > max_len:
        s = s[:max_len] + "\n..."
    return s


def build_messages_json(task_id, data):
    """Build a JS-friendly JSON array of messages for a trajectory."""
    meta = data["meta"]
    events = data["events"]
    messages = []
    turn_num = 0

    for evt in events:
        if evt.get("type") == "session":
            continue
        msg = evt.get("message", {})
        role = msg.get("role", "unknown")
        content = extract_text(msg.get("content", ""))
        tc = msg.get("toolCalls", [])
        tcid = msg.get("toolCallId", "")
        ts = evt.get("timestamp", "")

        if role == "user":
            turn_num += 1
            messages.append({
                "role": "user",
                "turn": turn_num,
                "text": content[:2000],
                "ts": ts,
            })
        elif role == "assistant":
            tools = []
            for c in tc:
                tn = c.get("serverName", c.get("tool", "?"))
                args = c.get("args", c.get("arguments", {}))
                tools.append({"name": tn, "args": fmt_json(args, 400)})

            privacy_tag = ""
            if "Hatch Trust Guidelines" in content:
                privacy_tag = "HARD_DENY"
            elif re.search(r"(consent|permission|ok to|cool to|go ahead)", content.lower()) and "?" in content:
                privacy_tag = "CONSENT_GATE"
            elif "minimum-necessary" in content.lower():
                privacy_tag = "MINIMIZE"
            elif "vault" in content.lower() and "encrypted" in content.lower():
                privacy_tag = "VAULT_REDIRECT"

            if not content.strip() and not tools:
                continue

            messages.append({
                "role": "assistant",
                "turn": turn_num,
                "text": content[:2000],
                "tools": tools,
                "privacy_tag": privacy_tag,
                "ts": ts,
            })
        elif role == "tool":
            messages.append({
                "role": "tool",
                "turn": turn_num,
                "text": content[:800],
                "call_id": tcid[:40],
                "ts": ts,
            })

    return messages


# Build all data
all_data = {}
for tid, data in trajectories.items():
    meta = data["meta"]
    v = meta.get("verification", {})
    all_data[tid] = {
        "meta": {
            "task_id": tid,
            "submission_id": meta.get("submission_id", ""),
            "verdict": v.get("verdict", "?"),
            "privacy_score": v.get("privacy_score"),
            "naturality_score": v.get("naturality_score"),
            "overall_score": v.get("overall_score"),
            "pii_level": meta.get("pii_level", "?"),
            "scenarios": meta.get("scenarios_covered", []),
            "skills_used": meta.get("skills_used", []),
            "decision_points": meta.get("decision_points", 0),
            "issues": v.get("issues", []),
            "rationale": v.get("rationale", ""),
        },
        "messages": build_messages_json(tid, data),
    }

data_json = json.dumps(all_data, default=str)
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Privacy SFT Trajectory Viewer</title>
<style>
:root {{
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --border: #30363d;
  --text: #c9d1d9; --text2: #8b949e; --text3: #f0f6fc;
  --blue: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922;
  --accent: #1f6feb;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }}

/* Header */
.top-bar {{ background: var(--bg2); padding: 20px 28px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }}
.top-bar h1 {{ font-size: 20px; color: var(--blue); font-weight: 600; }}
.top-bar .stats {{ display: flex; gap: 16px; align-items: center; font-size: 13px; color: var(--text2); }}
.stat-chip {{ background: var(--bg3); padding: 4px 12px; border-radius: 20px; }}
.stat-chip.pass {{ color: var(--green); border: 1px solid var(--green); }}

/* Layout */
.layout {{ display: flex; height: calc(100vh - 65px); }}

/* Sidebar */
.sidebar {{ width: 300px; min-width: 300px; background: var(--bg2); border-right: 1px solid var(--border); display: flex; flex-direction: column; }}
.sidebar-header {{ padding: 16px; border-bottom: 1px solid var(--border); }}
.sidebar-header label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text2); display: block; margin-bottom: 6px; }}
.dropdown {{ width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text3); padding: 10px 14px; border-radius: 8px; font-size: 14px; cursor: pointer; appearance: none; -webkit-appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238b949e' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 12px center; }}
.dropdown:focus {{ outline: none; border-color: var(--accent); }}
.dropdown option {{ background: var(--bg3); color: var(--text); }}

/* Info panel */
.info-panel {{ flex: 1; overflow-y: auto; padding: 16px; }}
.info-section {{ margin-bottom: 16px; }}
.info-section h3 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text2); margin-bottom: 8px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }}
.info-row {{ display: flex; justify-content: space-between; align-items: center; padding: 6px 0; font-size: 13px; }}
.info-row .label {{ color: var(--text2); }}
.info-row .value {{ color: var(--text3); font-weight: 500; }}

.verdict-badge {{ display: inline-block; padding: 3px 14px; border-radius: 16px; font-weight: 700; font-size: 12px; letter-spacing: 0.5px; }}
.verdict-badge.PASS {{ background: #238636; color: white; }}
.verdict-badge.FAIL {{ background: #da3633; color: white; }}
.verdict-badge.MINOR_ISSUES {{ background: #d29922; color: #0d1117; }}

.scenario-chip {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin: 2px; background: var(--bg); border: 1px solid var(--border); }}

.issue-card {{ padding: 8px 10px; margin-bottom: 6px; border-radius: 6px; font-size: 12px; line-height: 1.5; border-left: 3px solid; background: rgba(255,255,255,0.02); }}
.issue-card.minor {{ border-color: var(--yellow); }}
.issue-card.major {{ border-color: var(--red); }}
.issue-card.critical {{ border-color: var(--red); background: rgba(218,54,51,0.08); }}
.issue-sev {{ font-weight: 700; text-transform: uppercase; font-size: 10px; margin-right: 4px; }}

.rationale-box {{ background: var(--bg); padding: 10px 12px; border-radius: 6px; font-size: 12px; line-height: 1.6; color: var(--text2); border: 1px solid var(--border); }}

/* Main chat area */
.chat-area {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
.chat-toolbar {{ padding: 10px 20px; background: var(--bg2); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; font-size: 13px; }}
.chat-toolbar .filter-btn {{ background: var(--bg3); border: 1px solid var(--border); color: var(--text2); padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; transition: all 0.15s; }}
.chat-toolbar .filter-btn:hover {{ background: var(--border); }}
.chat-toolbar .filter-btn.active {{ background: var(--accent); border-color: var(--accent); color: white; }}
.msg-count {{ color: var(--text2); margin-left: auto; }}

.chat-scroll {{ flex: 1; overflow-y: auto; padding: 20px; }}

/* Messages */
.msg {{ display: flex; gap: 12px; margin-bottom: 12px; padding: 14px 16px; border-radius: 12px; border: 1px solid var(--border); animation: fadeIn 0.2s ease; }}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: translateY(0); }} }}

.msg.user {{ background: linear-gradient(135deg, #0d2240, #122a4e); border-color: #1f4070; }}
.msg.assistant {{ background: var(--bg2); }}
.msg.assistant.privacy {{ border-left: 3px solid var(--yellow); }}
.msg.tool {{ background: rgba(22,27,34,0.6); border-color: rgba(48,54,61,0.6); }}

.msg-icon {{ width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }}
.msg.user .msg-icon {{ background: #1f4070; }}
.msg.assistant .msg-icon {{ background: var(--bg3); }}
.msg.tool .msg-icon {{ background: var(--bg); }}

.msg-body {{ flex: 1; min-width: 0; }}
.msg-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
.msg-role {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }}
.msg.user .msg-role {{ color: var(--blue); }}
.msg.assistant .msg-role {{ color: var(--green); }}
.msg.tool .msg-role {{ color: var(--text2); }}
.msg-ts {{ font-size: 11px; color: var(--text2); }}
.msg-turn {{ font-size: 10px; color: var(--text2); background: var(--bg); padding: 1px 6px; border-radius: 4px; }}

.msg-text {{ font-size: 14px; line-height: 1.7; white-space: pre-wrap; word-wrap: break-word; }}

.privacy-badge {{ display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 5px; font-size: 11px; font-weight: 700; letter-spacing: 0.3px; margin-bottom: 8px; }}
.privacy-badge.HARD_DENY {{ background: #da3633; color: white; }}
.privacy-badge.CONSENT_GATE {{ background: var(--accent); color: white; }}
.privacy-badge.MINIMIZE {{ background: var(--yellow); color: #0d1117; }}
.privacy-badge.VAULT_REDIRECT {{ background: #8b5cf6; color: white; }}

.tool-block {{ background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 6px; }}
.tool-name {{ color: #79c0ff; font-family: 'SF Mono', Menlo, monospace; font-size: 13px; font-weight: 600; margin-bottom: 4px; }}
.tool-args {{ font-size: 11px; color: var(--text2); font-family: 'SF Mono', Menlo, monospace; white-space: pre-wrap; max-height: 120px; overflow-y: auto; }}

.tool-result-block {{ background: var(--bg); border-radius: 6px; padding: 8px 10px; font-size: 11px; font-family: 'SF Mono', Menlo, monospace; color: var(--text2); white-space: pre-wrap; max-height: 150px; overflow-y: auto; }}
.tool-result-id {{ font-size: 10px; color: var(--text2); margin-bottom: 4px; }}

.empty-state {{ display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--text2); }}
.empty-state .icon {{ font-size: 48px; margin-bottom: 12px; opacity: 0.4; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text2); }}
</style>
</head>
<body>

<div class="top-bar">
  <h1>Privacy SFT Trajectory Viewer</h1>
  <div class="stats">
    <span>Generated {now}</span>
    <span class="stat-chip pass">3/3 PASS</span>
  </div>
</div>

<div class="layout">
  <div class="sidebar">
    <div class="sidebar-header">
      <label>Select Trajectory</label>
      <select class="dropdown" id="traj-select" onchange="loadTrajectory(this.value)">
        <option value="" disabled>Choose a trajectory...</option>
      </select>
    </div>
    <div class="info-panel" id="info-panel">
      <div class="empty-state">
        <div class="icon">\U0001f50d</div>
        <p>Select a trajectory to view details</p>
      </div>
    </div>
  </div>

  <div class="chat-area">
    <div class="chat-toolbar" id="chat-toolbar" style="display:none">
      <span style="color:var(--text2);font-size:12px">Filter:</span>
      <button class="filter-btn active" data-filter="all" onclick="setFilter('all')">All</button>
      <button class="filter-btn" data-filter="user" onclick="setFilter('user')">\U0001f464 User</button>
      <button class="filter-btn" data-filter="assistant" onclick="setFilter('assistant')">\U0001f916 Assistant</button>
      <button class="filter-btn" data-filter="tool" onclick="setFilter('tool')">\u2699\ufe0f Tool</button>
      <button class="filter-btn" data-filter="privacy" onclick="setFilter('privacy')">\U0001f6e1 Privacy</button>
      <span class="msg-count" id="msg-count"></span>
    </div>
    <div class="chat-scroll" id="chat-scroll">
      <div class="empty-state">
        <div class="icon">\U0001f4ac</div>
        <p>Select a trajectory from the dropdown to view the conversation</p>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};
let currentFilter = 'all';
let currentMessages = [];

// Populate dropdown
const sel = document.getElementById('traj-select');
Object.keys(DATA).forEach((tid, i) => {{
  const d = DATA[tid];
  const opt = document.createElement('option');
  opt.value = tid;
  opt.textContent = tid + ' — ' + d.meta.pii_level + ' — ' + d.meta.verdict + ' — ' + d.messages.length + ' msgs';
  sel.appendChild(opt);
}});

// Auto-select first
sel.value = Object.keys(DATA)[0];
loadTrajectory(sel.value);

function loadTrajectory(tid) {{
  const d = DATA[tid];
  if (!d) return;
  currentMessages = d.messages;
  renderInfo(d.meta);
  renderMessages();
  document.getElementById('chat-toolbar').style.display = 'flex';
}}

function renderInfo(m) {{
  const panel = document.getElementById('info-panel');
  let issuesHtml = '';
  (m.issues || []).forEach(iss => {{
    issuesHtml += `<div class="issue-card ${{iss.severity}}"><span class="issue-sev" style="color:${{iss.severity==='minor'?'var(--yellow)':'var(--red)'}}">${{iss.severity}}</span>${{esc(iss.description || '')}}</div>`;
  }});

  let scenariosHtml = (m.scenarios || []).map(s => `<span class="scenario-chip">${{s}}</span>`).join('');
  let skillsHtml = (m.skills_used || []).map(s => `<span class="scenario-chip">${{s}}</span>`).join('') || '<span style="color:var(--text2)">—</span>';

  panel.innerHTML = `
    <div class="info-section">
      <h3>Verdict</h3>
      <div style="text-align:center;padding:8px 0">
        <span class="verdict-badge ${{m.verdict}}">${{m.verdict}}</span>
      </div>
      <div class="info-row"><span class="label">Task</span><span class="value">${{m.task_id}}</span></div>
      <div class="info-row"><span class="label">Submission</span><span class="value" style="font-size:11px;font-family:monospace">${{m.submission_id.substring(0,18)}}...</span></div>
      <div class="info-row"><span class="label">PII Level</span><span class="value">${{m.pii_level}}</span></div>
      <div class="info-row"><span class="label">Decision Points</span><span class="value">${{m.decision_points}}</span></div>
    </div>
    <div class="info-section">
      <h3>Scenarios</h3>
      <div style="padding:4px 0">${{scenariosHtml}}</div>
    </div>
    <div class="info-section">
      <h3>Skills Used</h3>
      <div style="padding:4px 0">${{skillsHtml}}</div>
    </div>
    ${{m.issues && m.issues.length ? '<div class="info-section"><h3>Issues (' + m.issues.length + ')</h3>' + issuesHtml + '</div>' : ''}}
    <div class="info-section">
      <h3>Rationale</h3>
      <div class="rationale-box">${{esc(m.rationale || 'No rationale provided.')}}</div>
    </div>
  `;
}}

function renderMessages() {{
  const container = document.getElementById('chat-scroll');
  let filtered = currentMessages;

  if (currentFilter === 'privacy') {{
    filtered = currentMessages.filter(m => m.privacy_tag);
  }} else if (currentFilter !== 'all') {{
    filtered = currentMessages.filter(m => m.role === currentFilter);
  }}

  document.getElementById('msg-count').textContent = filtered.length + ' / ' + currentMessages.length + ' messages';

  if (filtered.length === 0) {{
    container.innerHTML = '<div class="empty-state"><div class="icon">\U0001f50d</div><p>No messages match this filter</p></div>';
    return;
  }}

  let html = '';
  filtered.forEach((m, i) => {{
    const isPrivacy = m.privacy_tag ? ' privacy' : '';
    const icon = m.role === 'user' ? '\U0001f464' : m.role === 'assistant' ? '\U0001f916' : '\u2699\ufe0f';
    const turnBadge = m.turn ? `<span class="msg-turn">Turn ${{m.turn}}</span>` : '';

    let body = '';

    if (m.privacy_tag) {{
      body += `<div class="privacy-badge ${{m.privacy_tag}}">${{m.privacy_tag.replace('_', ' ')}}</div>`;
    }}

    if (m.tools && m.tools.length) {{
      m.tools.forEach(t => {{
        body += `<div class="tool-block"><div class="tool-name">\U0001f527 ${{esc(t.name)}}</div><div class="tool-args">${{esc(t.args)}}</div></div>`;
      }});
    }}

    if (m.role === 'tool') {{
      body += `<div class="tool-result-id">Result for ${{esc(m.call_id || '?')}}</div><div class="tool-result-block">${{esc(m.text)}}</div>`;
    }} else if (m.text) {{
      body += `<div class="msg-text">${{esc(m.text)}}</div>`;
    }}

    html += `
      <div class="msg ${{m.role}}${{isPrivacy}}">
        <div class="msg-icon">${{icon}}</div>
        <div class="msg-body">
          <div class="msg-header">
            <span class="msg-role">${{m.role}}</span>
            ${{turnBadge}}
          </div>
          ${{body}}
        </div>
      </div>`;
  }});

  container.innerHTML = html;
  container.scrollTop = 0;
}}

function setFilter(f) {{
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.toggle('active', b.dataset.filter === f));
  renderMessages();
}}

function esc(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}
</script>
</body></html>"""

DEST.write_text(page)
print(f"Written to {DEST} ({len(page)//1024} KB)")
