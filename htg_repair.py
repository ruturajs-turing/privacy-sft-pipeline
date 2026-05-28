#!/usr/bin/env python3
"""htg_repair.py — Universal HTG-compliance repair pass for OpenClaw trajectories.

Takes an OpenClaw JSONL session export, detects Hatch Trust Guidelines (HTG)
violations, and uses Claude Opus 4.7 to regenerate the offending turns into
compliant, natural-reading dialogue — inserting any user turns / tool calls
needed to keep the corrected flow coherent.

DATA INTEGRITY: every regenerated or inserted event carries a `provenance`
block (origin=regenerated, source id, model, ruleset, timestamp). This tool
deliberately does NOT strip or disguise that provenance — repaired golden
trajectories must remain auditable by whoever consumes the dataset.

Usage:
    python htg_repair.py <session.jsonl> [--out <file>] [--dry-run]
    python htg_repair.py <dir-or-glob> --batch

Requires ANTHROPIC_API_KEY in env / .env (uses claude-opus-4-7).
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("htg_repair")

# ---------------------------------------------------------------------------
# Config (mirrors privacy-sft-pipeline/config.py; kept self-contained)
# ---------------------------------------------------------------------------
REPAIR_MODEL = "claude-opus-4-7"
MAX_TOKENS = 8000
BASE_DIR = Path(__file__).resolve().parent
HATCH_REFUSAL = (BASE_DIR / "data" / "hatch_refusal.txt").read_text().strip() \
    if (BASE_DIR / "data" / "hatch_refusal.txt").exists() \
    else ("Under Hatch Trust Guidelines, I cannot retain this classification of "
           "sensitive data. Please provide it again for the current task.")

REDACT_PLACEHOLDER = "<REDACTED_L4_SECRET>"

# Persistence tools that hard-deny at L3/L4 (Gate 4a)
PERSISTENCE_WRITE_TOOLS = {
    "memory_write", "active_memory_write", "active_memory_set", "wiki_apply",
}

# ---------------------------------------------------------------------------
# Secret / high-entropy detection (heuristic candidate finder)
# ---------------------------------------------------------------------------
SECRET_PREFIXES = ("sk-", "sk_", "ntn_", "secret_", "ghp_", "gho_", "xoxb-",
                   "xoxp-", "AKIA", "AIza", "pk_live", "rk_live", "Bearer ")
SECRET_KEY_HINT = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|private[_-]?key|access[_-]?key)",
    re.IGNORECASE)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def find_secrets(text: str) -> list[str]:
    """Return substrings that look like live credentials/secrets."""
    if not isinstance(text, str) or not text:
        return []
    found: set[str] = set()
    for tok in re.findall(r"[A-Za-z0-9_\-\.]{20,}", text):
        if tok.startswith(SECRET_PREFIXES):
            found.add(tok)
        elif len(tok) >= 32 and _shannon_entropy(tok) >= 3.5:
            # high-entropy long token — likely a key
            found.add(tok)
    for pref in SECRET_PREFIXES:
        for m in re.finditer(re.escape(pref) + r"[A-Za-z0-9_\-]+", text):
            found.add(m.group(0))
    return sorted(found, key=len, reverse=True)


_SECRET_FILE = r"[^\s|;&)\"']*(?:api[_-]?key|secret|token|credential|\.env|id_rsa)[^\s|;&)\"']*"


def _exec_prints_secret(cmd: str) -> bool:
    """True only if the shell command would emit a secret to stdout.

    Flags bare `cat/less/head/tail <secretfile>` (output goes to the terminal)
    and `echo/printf $SECRETVAR`. Does NOT flag existence checks
    (`[ -s file ]`, `test -f`), `$(cat file)` capture, or use in an auth header.
    """
    if not isinstance(cmd, str) or not cmd:
        return False
    for m in re.finditer(r"(cat|less|head|tail|bat)\s+(" + _SECRET_FILE + r")", cmd, re.I):
        prefix = cmd[max(0, m.start() - 3):m.start()]
        if "$(" in prefix or "`" in prefix:      # command substitution -> captured
            continue
        tail = cmd[m.end():m.end() + 40]
        if re.match(r"\s*\|\s*(wc|sha\d*sum|md5|md5sum|cksum|stat)\b", tail):
            continue                              # piped into a non-revealing summarizer
        return True
    # echo/printf of a variable that holds a secret
    if re.search(r"\b(echo|printf)\b[^\n;&|]*\$\{?[A-Za-z_]*(KEY|SECRET|TOKEN|PASS|CRED)",
                 cmd, re.I):
        return True
    return False


def scrub_secrets(text: str, secrets: set[str]) -> str:
    """Replace any known secret value with the redaction placeholder."""
    if not isinstance(text, str):
        return text
    for sec in sorted(secrets, key=len, reverse=True):
        if sec and sec in text:
            text = text.replace(sec, REDACT_PLACEHOLDER)
    return text


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------
def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def msg_role(ev: dict) -> str | None:
    if ev.get("type") != "message":
        return None
    m = ev.get("message")
    return m.get("role") if isinstance(m, dict) else None


def blocks(ev: dict) -> list[dict]:
    m = ev.get("message")
    c = m.get("content") if isinstance(m, dict) else None
    return c if isinstance(c, list) else []


# ---------------------------------------------------------------------------
# Heuristic violation detector (universal; gives the LLM grounded hints)
# ---------------------------------------------------------------------------
def detect_violations(events: list[dict]) -> tuple[list[dict], set[str]]:
    """Return (violations, all_secret_values).

    Each violation: {line, id, code, severity, detail}.
    """
    violations: list[dict] = []
    all_secrets: set[str] = set()

    for i, ev in enumerate(events, start=1):
        rid = ev.get("id", "")
        role = msg_role(ev)
        for b in blocks(ev):
            bt = b.get("type")
            text = b.get("text") or b.get("thinking") or ""
            secs = find_secrets(text)
            if secs:
                all_secrets.update(secs)

            # secret retained in assistant thinking
            if role == "assistant" and bt == "thinking" and secs:
                violations.append(dict(
                    line=i, id=rid, code="secret_in_thinking", severity="critical",
                    detail=f"Assistant thinking contains a credential-like value "
                           f"({len(secs)} token(s)). L4 secrets must never be retained "
                           f"in reasoning."))

            # secret echoed in any assistant text
            if role == "assistant" and bt == "text" and secs:
                violations.append(dict(
                    line=i, id=rid, code="secret_in_assistant_text", severity="high",
                    detail="Assistant message echoes a credential-like value."))

            # tool result printing a secret
            if role == "toolResult" and bt == "text" and secs:
                violations.append(dict(
                    line=i, id=rid, code="secret_in_tool_result", severity="high",
                    detail="Tool result prints a raw secret back into the transcript."))

            # tool calls
            if bt == "toolCall":
                name = b.get("name", "")
                args_str = json.dumps(b.get("arguments", {}))
                arg_secs = find_secrets(args_str)
                if arg_secs:
                    all_secrets.update(arg_secs)
                # Gate 4a: T1 persistence write of L4 secret
                if name in PERSISTENCE_WRITE_TOOLS and arg_secs:
                    violations.append(dict(
                        line=i, id=rid, code="l4_persistence_write", severity="critical",
                        detail=f"`{name}` persists an L4 secret — Gate 4a hard-deny."))
                # shell command that PRINTS a secret to stdout (not mere existence
                # checks or $(...) capture into an auth header)
                if name == "exec":
                    cmd = str(b.get("arguments", {}).get("command", ""))
                    if _exec_prints_secret(cmd):
                        violations.append(dict(
                            line=i, id=rid, code="secret_exec_echo", severity="high",
                            detail=f"`exec` command prints a secret to stdout: {cmd[:120]}"))

            # bogus refusal: emits Hatch refusal with no sensitive op in the turn
            if role == "assistant" and bt == "text":
                t = b.get("text", "")
                if "Hatch Trust Guidelines" in t or "Hatch Trust" in t:
                    has_sensitive_op = any(
                        bb.get("type") == "toolCall" and bb.get("name") in PERSISTENCE_WRITE_TOOLS
                        for bb in blocks(ev))
                    if not has_sensitive_op:
                        violations.append(dict(
                            line=i, id=rid, code="bogus_refusal", severity="medium",
                            detail="Emits an HTG refusal with no actual L3/L4 persistence "
                                   "operation in the turn — over-refusal."))
    return violations, all_secrets


# ---------------------------------------------------------------------------
# LLM repair
# ---------------------------------------------------------------------------
REPAIR_SYSTEM = f"""You are an HTG-compliance trajectory repairer for the OpenClaw Privacy SFT project.

You are given an OpenClaw session as a numbered list of events, plus a list of detected Hatch Trust Guidelines (HTG) violations. Your job: regenerate ONLY the violating turns (and insert any user turns / tool calls strictly needed to make the corrected flow coherent) so the assistant demonstrates perfect HTG compliance — while reading like a natural, real conversation, NOT a template.

## HTG decision model (5-step authorization)
1. Classify L = max data level across the operation. L4=secrets/keys/passwords/PANs/SSNs. L3=PHI, salary, precise location, etc. L2=names/emails/phones. L1=device/IP. L0=public.
2. Provenance: L3+ data is only admissible if it came from the current user turn (or an explicit approval immediately prior). Else HALT and ask the user to re-provide it.
3. Tool select: lowest tier capable.
4. Gate:
   - 4a HARD DENY: tier T1 AND L>=3 AND tool in {{memory_write, active_memory_write, active_memory_set, wiki_apply}}. Do NOT call, do NOT paraphrase the secret. Respond verbatim: "{HATCH_REFUSAL}"
   - 4b CONFIRM: T2 AND L>=3 — disclose + await approval.
   - 4c CONFIRM: T3 AND L>=2 — disclose 3P destination + await approval.
   - 4d ALLOW otherwise.
5. Execute.

## Hard rules for your rewrite
- An L4 secret (API key, password, token, private key) must NEVER appear in assistant thinking, assistant text, tool-call arguments that persist it, or any tool result. If you need to reference it, use {REDACT_PLACEHOLDER} or describe it ("the key you pasted").
- NEVER run shell commands (cat/echo/etc.) that print a secret sink back to the transcript. Replace such checks with a non-revealing alternative (e.g. test the key against the live API and report only pass/fail), or drop them.
- Do NOT invent HTG refusals for turns that have no L3/L4 persistence operation. Over-refusal is itself a defect — fix `bogus_refusal` by removing the refusal and just doing the (benign) task.
- When a secret was exposed, the compliant assistant guidance is: don't persist it; advise the user to revoke/rotate the exposed credential and provide a fresh one in-turn when actually needed.
- Keep workspace paths, tool names, IDs context, and conversational intent consistent with the surrounding (non-violating) turns. Preserve everything that wasn't a violation.
- TOOL-CALL PAIRING: every toolCall event you emit MUST be immediately followed, in the SAME operation's `events` list, by its toolResult event (role "toolResult"). NEVER narrate a tool outcome (e.g. "Notion returned 401") in an assistant turn without first emitting the toolResult event that produced it. A toolCall with no following toolResult is invalid.
- Write natural dialogue. Vary phrasing. The user turns you insert should sound like a real person (this persona is "Lauren" unless context says otherwise).

## Output format (STRICT JSON, no prose)
Return a JSON object: {{"operations": [ ... ]}}
Each operation is one of:
  {{"action": "replace", "target_line": <int>, "events": [<event>, ...]}}
  {{"action": "insert_after", "target_line": <int>, "events": [<event>, ...]}}
  {{"action": "delete", "target_line": <int>}}
Each <event> is a full OpenClaw event object WITHOUT id/parentId/timestamp (the harness assigns those). Use this shape:
  {{"type":"message","message":{{"role":"user|assistant|toolResult","content":[ ...blocks... ]}}}}
Blocks: {{"type":"text","text":"..."}} | {{"type":"thinking","thinking":"..."}} | {{"type":"toolCall","name":"...","arguments":{{...}}}}
Only emit operations for lines that need to change. Do not restate unchanged lines.
"""


def compact_transcript(events: list[dict], secrets: set[str]) -> str:
    """Render a numbered, secret-scrubbed view for the LLM."""
    lines = []
    for i, ev in enumerate(events, start=1):
        t = ev.get("type")
        if t != "message":
            data = scrub_secrets(json.dumps(ev.get("data", ev.get("content", ""))), secrets)
            lines.append(f"[{i}] event:{t} {data[:160]}")
            continue
        role = msg_role(ev)
        for b in blocks(ev):
            bt = b.get("type")
            if bt == "text":
                lines.append(f"[{i}] {role}/text: {scrub_secrets(b.get('text',''), secrets)}")
            elif bt == "thinking":
                lines.append(f"[{i}] {role}/thinking: {scrub_secrets(b.get('thinking',''), secrets)}")
            elif bt == "toolCall":
                args = scrub_secrets(json.dumps(b.get("arguments", {})), secrets)
                lines.append(f"[{i}] {role}/toolCall {b.get('name')}: {args}")
            else:
                lines.append(f"[{i}] {role}/{bt}")
    return "\n".join(lines)


async def request_repairs(events: list[dict], violations: list[dict],
                          secrets: set[str]) -> list[dict]:
    import anthropic  # imported lazily so --dry-run works without the SDK
    client = anthropic.AsyncAnthropic()
    transcript = compact_transcript(events, secrets)
    vio_text = "\n".join(
        f"- line {v['line']} (id {v['id']}) [{v['severity']}] {v['code']}: {v['detail']}"
        for v in violations)
    user_msg = (f"SESSION TRANSCRIPT (secrets already redacted to {REDACT_PLACEHOLDER}):\n"
                f"{transcript}\n\n"
                f"DETECTED VIOLATIONS:\n{vio_text}\n\n"
                f"Return the operations JSON that repairs every violation.")
    resp = await client.messages.create(
        model=REPAIR_MODEL, max_tokens=MAX_TOKENS,
        system=REPAIR_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"LLM returned no JSON object:\n{raw[:500]}")
    return json.loads(m.group(0)).get("operations", [])


# ---------------------------------------------------------------------------
# Apply operations + re-link the event stream
# ---------------------------------------------------------------------------
_ID_SEQ = 0


def _new_id() -> str:
    """Deterministic 8-hex id (avoids Math.random/Date prohibitions)."""
    global _ID_SEQ
    _ID_SEQ += 1
    return f"rpr{_ID_SEQ:05d}"


def provenance(source_id: str | None) -> dict:
    return {
        "origin": "regenerated",
        "source_id": source_id,
        "model": REPAIR_MODEL,
        "ruleset": "HTG/New-rules-5-step",
        "tool": "htg_repair.py",
    }


def _scrub_event(ev: dict, secrets: set[str]) -> None:
    """In-place: remove any known secret value from an event's content/args."""
    if ev.get("type") != "message":
        return
    for b in ev.get("message", {}).get("content", []) or []:
        for fld in ("text", "thinking"):
            if isinstance(b.get(fld), str):
                b[fld] = scrub_secrets(b[fld], secrets)
        if b.get("type") == "toolCall":
            b["arguments"] = json.loads(scrub_secrets(json.dumps(b.get("arguments", {})), secrets))


def pair_dangling_tool_calls(events: list[dict]) -> list[dict]:
    """Insert a synthesized toolResult after any toolCall that lacks one.

    The LLM occasionally emits a toolCall without its paired toolResult (then
    narrates the outcome in the next assistant turn), which leaves a dangling
    call and adjacent assistant messages. This mirrors `drop_orphan_tool_results`
    (which handles the opposite case) and guarantees every call has a result.
    Synthesized results are tagged `origin=synthesized_tool_result` so they stay
    auditable and distinct from regenerated content.
    """
    resolved = {
        ev["message"]["toolCallId"]
        for ev in events
        if msg_role(ev) == "toolResult"
        and isinstance(ev.get("message"), dict)
        and ev["message"].get("toolCallId")
    }
    out: list[dict] = []
    for ev in events:
        out.append(ev)
        for b in blocks(ev):
            if b.get("type") != "toolCall":
                continue
            cid = b.get("id")
            if not cid or cid in resolved:
                continue
            synth = {
                "type": "message",
                "id": _new_id(),
                "parentId": ev.get("id"),
                "message": {
                    "role": "toolResult",
                    "toolCallId": cid,
                    "toolName": b.get("name"),
                    "content": [{"type": "text",
                                 "text": "(tool result reconstructed during HTG "
                                         "repair; see the assistant summary that "
                                         "follows)"}],
                },
                "provenance": {**provenance(ev.get("id")),
                               "origin": "synthesized_tool_result"},
            }
            out.append(synth)
            resolved.add(cid)
    return out


def _stamp_replacement(orig: dict, new: dict, secrets: set[str]) -> dict:
    """Build a replacement event that inherits the original's id/linkage so the
    call<->result pairing and parent chain stay intact."""
    ev = deepcopy(new)
    ev["id"] = orig.get("id", _new_id())          # reuse original id -> no orphans
    if orig.get("parentId") is not None:
        ev["parentId"] = orig["parentId"]
    if orig.get("timestamp") is not None:
        ev["timestamp"] = orig["timestamp"]
    omsg = orig.get("message", {}) if isinstance(orig.get("message"), dict) else {}
    nmsg = ev.get("message", {}) if isinstance(ev.get("message"), dict) else {}
    # toolResult linkage: keep the original toolCallId / toolName
    if nmsg.get("role") == "toolResult":
        if omsg.get("toolCallId"):
            nmsg["toolCallId"] = omsg["toolCallId"]
        if omsg.get("toolName"):
            nmsg["toolName"] = omsg["toolName"]
    # toolCall block ids: reuse original block ids positionally
    o_calls = [b for b in omsg.get("content", []) if b.get("type") == "toolCall"]
    n_calls = [b for b in nmsg.get("content", []) if b.get("type") == "toolCall"]
    for idx, nb in enumerate(n_calls):
        nb["id"] = o_calls[idx]["id"] if idx < len(o_calls) and o_calls[idx].get("id") \
            else f"call_{_new_id()}"
    _scrub_event(ev, secrets)
    ev["provenance"] = provenance(orig.get("id"))
    return ev


def _stamp_insert(new: dict, source_id: str | None, secrets: set[str],
                  last_call_id: list[str | None]) -> dict:
    """Build a freshly-inserted event with new ids; link toolResults to the
    most recent toolCall id seen in the output stream."""
    ev = deepcopy(new)
    ev["id"] = _new_id()
    msg = ev.get("message", {}) if isinstance(ev.get("message"), dict) else {}
    for b in msg.get("content", []) or []:
        if b.get("type") == "toolCall":
            b["id"] = f"call_{ev['id']}"
            last_call_id[0] = b["id"]
    if msg.get("role") == "toolResult" and not msg.get("toolCallId") and last_call_id[0]:
        msg["toolCallId"] = last_call_id[0]
    _scrub_event(ev, secrets)
    ev["provenance"] = provenance(source_id)
    return ev


def apply_operations(events: list[dict], ops: list[dict], secrets: set[str]) -> list[dict]:
    by_line: dict[int, list[dict]] = {}
    replace: dict[int, list[dict]] = {}
    delete: set[int] = set()
    for op in ops:
        line = op.get("target_line")
        action = op.get("action")
        if action == "delete":
            delete.add(line)
        elif action == "replace":
            replace[line] = op.get("events", [])
        elif action == "insert_after":
            by_line.setdefault(line, []).extend(op.get("events", []))

    out: list[dict] = []
    last_call_id: list[str | None] = [None]
    for i, ev in enumerate(events, start=1):
        src_id = ev.get("id")
        if i in delete:
            log.info("  delete line %d (%s)", i, src_id)
            continue
        if i in replace:
            log.info("  replace line %d (%s) -> %d event(s)", i, src_id, len(replace[i]))
            news = replace[i]
            # first replacement inherits the original's structural slot
            for j, ne in enumerate(news):
                if j == 0:
                    rep = _stamp_replacement(ev, ne, secrets)
                else:
                    rep = _stamp_insert(ne, src_id, secrets, last_call_id)
                for b in rep.get("message", {}).get("content", []) or []:
                    if b.get("type") == "toolCall" and b.get("id"):
                        last_call_id[0] = b["id"]
                out.append(rep)
        else:
            _scrub_event(ev, secrets)              # scrub passed-through originals too
            for b in ev.get("message", {}).get("content", []) or []:
                if b.get("type") == "toolCall" and b.get("id"):
                    last_call_id[0] = b["id"]
            out.append(ev)
        if i in by_line:
            log.info("  insert after line %d -> %d event(s)", i, len(by_line[i]))
            for ne in by_line[i]:
                out.append(_stamp_insert(ne, src_id, secrets, last_call_id))

    # pair any toolCall the LLM left without a result (prevents dangling calls
    # and the consecutive-assistant warning they cause)
    before_pair = len(out)
    out = pair_dangling_tool_calls(out)
    if len(out) != before_pair:
        log.info("  synthesized %d tool result(s) for dangling call(s)",
                 len(out) - before_pair)

    # deterministic structural normalization (pipeline's own SFT-ready pass):
    # drops empty/orphan turns, collapses consecutive assistants, repairs parents.
    # normalize_event_stream is NOT idempotent — it collapses BEFORE dropping
    # orphan tool results, so an orphan sitting between two assistant turns
    # prevents their merge until a later pass. Iterate to a fixpoint.
    try:
        from trajectory_structure import normalize_event_stream
        start = len(out)
        for _ in range(5):
            before = len(out)
            out = normalize_event_stream(out)
            if len(out) == before:
                break
        if len(out) != start:
            log.info("  normalized stream %d -> %d events", start, len(out))
    except Exception as e:  # noqa: BLE001
        log.debug("normalize skipped: %s", e)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
async def repair_file(path: Path, out: Path | None, dry_run: bool) -> dict:
    events = load_events(path)
    violations, secrets = detect_violations(events)
    log.info("%s: %d events, %d violation(s), %d secret value(s)",
             path.name, len(events), len(violations), len(secrets))
    for v in violations:
        log.info("  - L%s %s/%s: %s", v["line"], v["severity"], v["code"], v["detail"])

    report = {"file": str(path), "violations": violations,
              "secret_count": len(secrets), "operations": []}

    if not violations:
        log.info("  clean — nothing to repair")
        return report
    if dry_run:
        log.info("  [dry-run] would request %d-violation repair from %s",
                 len(violations), REPAIR_MODEL)
        return report

    ops = await request_repairs(events, violations, secrets)
    report["operations"] = ops
    repaired = apply_operations(events, ops, secrets)

    # structural validation if the pipeline module is importable
    try:
        from trajectory_structure import validate_event_stream
        issues = validate_event_stream(repaired)
        report["structure_issues"] = issues
        if issues:
            log.warning("  %d structural issue(s) after repair", len(issues))
    except Exception as e:  # noqa: BLE001
        log.debug("structure validation skipped: %s", e)

    # residual self-check: re-run detection + secret scan on the repaired stream
    residual, residual_secrets = detect_violations(repaired)
    repaired_blob = "\n".join(json.dumps(e) for e in repaired)
    leaked = sorted(s for s in secrets if s in repaired_blob)
    report["residual_violations"] = residual
    report["leaked_secrets"] = len(leaked)
    report["repaired_event_count"] = len(repaired)
    report["provenance_tagged"] = sum(1 for e in repaired if "provenance" in e)
    if leaked:
        log.error("  ✗ %d secret value(s) STILL present after repair", len(leaked))
    if residual:
        log.warning("  ✗ %d residual violation(s) remain after repair:", len(residual))
        for v in residual:
            log.warning("    - L%s %s/%s: %s", v["line"], v["severity"], v["code"], v["detail"])
    if not leaked and not residual:
        log.info("  ✓ clean: 0 residual violations, 0 leaked secrets")

    out = out or path.with_name(path.stem + "_repaired.jsonl")
    out.write_text(repaired_blob + "\n")
    out.with_suffix(".repair_report.json").write_text(json.dumps(report, indent=2))
    log.info("  wrote %s (%d events, %d provenance-tagged) + report",
             out.name, len(repaired), report["provenance_tagged"])
    return report


async def main_async(args) -> None:
    targets: list[Path] = []
    p = Path(args.path)
    if args.batch:
        if p.is_dir():
            targets = sorted(p.rglob("*.jsonl"))
        else:
            targets = [Path(x) for x in glob.glob(args.path, recursive=True)]
        targets = [t for t in targets if not t.stem.endswith("_repaired")]
    else:
        targets = [p]
    log.info("repairing %d file(s) with %s", len(targets), REPAIR_MODEL)
    for t in targets:
        try:
            await repair_file(t, Path(args.out) if args.out else None, args.dry_run)
        except Exception as e:  # noqa: BLE001
            log.error("FAILED %s: %s", t, e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Universal HTG-compliance trajectory repair (Opus 4.7).")
    ap.add_argument("path", help="session .jsonl file, or dir/glob with --batch")
    ap.add_argument("--out", help="output path (single-file mode)")
    ap.add_argument("--batch", action="store_true", help="treat path as dir/glob")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect + report only; no LLM call, no write")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
