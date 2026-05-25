"""Tool tier registry — maps all OpenClaw skills/tools to trust tiers with data level caps.

Based on Appendix A of the OpenClaw Privacy Proposal. Each tool has:
- tier: T1 (local), T2 (1P cloud), T3 (3P API)
- max_data_level: highest data sensitivity level the tool can safely handle
- distractor_pool: similar tools from OTHER tiers (for COVERT-style augmentation)
"""

from __future__ import annotations
from typing import Literal, Optional

TierName = Literal["T1", "T2", "T3"]

TIER_MAX_LEVELS = {
    "T1": "L4",  # Local tools can handle all data
    "T2": "L2",  # 1P Cloud: L2 by default; L3/L4 require explicit consent
    "T3": "L1",  # 3P API: L1 only; L2+ strictly prohibited without exception
}

LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}


def level_exceeds(data_level: str, tool_tier: str) -> bool:
    """Return True if data_level exceeds what the tool tier allows by default."""
    max_level = TIER_MAX_LEVELS.get(tool_tier, "L1")
    return LEVEL_ORDER.get(data_level, 0) > LEVEL_ORDER.get(max_level, 0)


TOOL_REGISTRY: dict[str, dict] = {
    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1 — Local (safe for all data up to L4)
    # ─────────────────────────────────────────────────────────────────────────
    "nano-pdf": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Local PDF text and metadata extraction",
        "distractor_pool": ["enterprise-inference", "web_fetch"],
    },
    "surya": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Local OCR and document layout analysis",
        "distractor_pool": ["enterprise-inference", "web_fetch"],
    },
    "openai-whisper": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Local Whisper model transcription",
        "distractor_pool": ["openai-whisper-api", "enterprise-inference"],
    },
    "word-docx": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Create/edit .docx files locally",
        "distractor_pool": ["notion", "gog"],
    },
    "excel-xlsx": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Read/write .xlsx spreadsheets locally",
        "distractor_pool": ["gog", "notion"],
    },
    "powerpoint-pptx": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Create/edit .pptx presentations locally",
        "distractor_pool": ["gog", "notion"],
    },
    "docker-essentials": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Manage local Docker containers",
        "distractor_pool": ["github", "clawhub"],
    },
    "data-analysis": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Statistical analysis with local Python stack",
        "distractor_pool": ["enterprise-inference", "stock-analysis"],
    },
    "automation-workflows": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Multi-step automation sequences in-process",
        "distractor_pool": ["slack", "notion", "trello"],
    },
    "productivity": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Local task and notes management",
        "distractor_pool": ["notion", "trello"],
    },
    "ontology": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Build and query local ontology graphs",
        "distractor_pool": ["enterprise-rag", "web_search"],
    },
    "self-improving": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Agent self-refinement loop (local)",
        "distractor_pool": ["enterprise-inference"],
    },
    "obsidian": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Read/write Obsidian vault notes locally",
        "distractor_pool": ["notion", "gog"],
    },
    "mcporter": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Local MCP server port management",
        "distractor_pool": ["api-gateway"],
    },
    "humanizer": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Post-process agent output for tone (local)",
        "distractor_pool": ["enterprise-inference"],
    },
    "markdown-converter": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Convert between formats locally",
        "distractor_pool": ["gog", "notion"],
    },
    "skill-creator": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Scaffold and register new skills locally",
        "distractor_pool": ["clawhub", "github"],
    },
    "ui-ux-pro-max": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Generate UI mockups locally",
        "distractor_pool": ["enterprise-inference"],
    },
    "marketing-mode": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Marketing strategy and copywriting (local)",
        "distractor_pool": ["web_search", "enterprise-inference"],
    },
    "self-reflection": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Structured reflection and durable memory (local)",
        "distractor_pool": ["enterprise-rag", "notion"],
    },
    "language-learning": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Conversational language tutor (local)",
        "distractor_pool": ["enterprise-inference", "web_search"],
    },
    "cfo": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Financial planning and guidance (local)",
        "distractor_pool": ["stock-analysis", "enterprise-inference"],
    },
    "health": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Wellness guidance with safety boundaries (local)",
        "distractor_pool": ["web_search", "enterprise-inference"],
    },
    "relationship-skills": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Communication and relationship tools (local)",
        "distractor_pool": ["enterprise-inference", "slack"],
    },
    "workout": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Gym session tracking (local)",
        "distractor_pool": ["web_search", "notion"],
    },
    "healthcheck": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Host security auditing (local)",
        "distractor_pool": ["github", "web_search"],
    },
    "mechanic": {
        "tier": "T1", "max_data_level": "L4",
        "description": "Vehicle maintenance tracker (local)",
        "distractor_pool": ["web_search", "goplaces"],
    },
    # Linux/OpenClaw native tools
    "read": {"tier": "T1", "max_data_level": "L4", "description": "Read file", "distractor_pool": ["enterprise-rag", "gog"]},
    "write": {"tier": "T1", "max_data_level": "L4", "description": "Write file", "distractor_pool": ["notion", "gog"]},
    "exec": {"tier": "T1", "max_data_level": "L4", "description": "Execute shell command", "distractor_pool": ["github", "api-gateway"]},
    "memory_search": {"tier": "T1", "max_data_level": "L4", "description": "Search agent memory", "distractor_pool": ["enterprise-rag", "web_search"]},
    "memory_write": {"tier": "T1", "max_data_level": "L2", "description": "Write to persistent memory (L2 max for privacy)", "distractor_pool": ["enterprise-rag", "notion"]},
    "get": {"tier": "T1", "max_data_level": "L4", "description": "MCP vault get(key) -> {value}", "distractor_pool": ["memory_search", "enterprise-vault"]},
    "set": {"tier": "T1", "max_data_level": "L4", "description": "MCP vault set(key,val) -> {ok}", "distractor_pool": ["memory_write", "enterprise-vault"]},
    "delete": {"tier": "T1", "max_data_level": "L4", "description": "MCP vault delete(key) -> {ok}", "distractor_pool": ["memory_write", "enterprise-vault"]},
    "vault_get": {"tier": "T1", "max_data_level": "L4", "description": "Read from encrypted vault", "distractor_pool": ["memory_search", "enterprise-vault"]},
    "vault_set": {"tier": "T1", "max_data_level": "L4", "description": "Write to encrypted vault", "distractor_pool": ["memory_write", "enterprise-vault"]},
    "vault_delete": {"tier": "T1", "max_data_level": "L4", "description": "Delete from encrypted vault", "distractor_pool": ["memory_write", "enterprise-vault"]},
    "cron": {"tier": "T1", "max_data_level": "L4", "description": "Schedule tasks", "distractor_pool": ["enterprise-calendar", "trello"]},
    "sessions": {"tier": "T1", "max_data_level": "L4", "description": "Spawn subagent sessions", "distractor_pool": ["slack", "enterprise-inference"]},

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 2 — 1P Cloud (safe up to L2; L3/L4 require explicit consent)
    # ─────────────────────────────────────────────────────────────────────────
    "enterprise-mail": {
        "tier": "T2", "max_data_level": "L2",
        "description": "Send/retrieve enterprise email",
        "distractor_pool": ["slack", "notion", "nano-pdf"],
    },
    "enterprise-calendar": {
        "tier": "T2", "max_data_level": "L2",
        "description": "Manage calendar events",
        "distractor_pool": ["caldav-calendar", "cron", "trello"],
    },
    "enterprise-rag": {
        "tier": "T2", "max_data_level": "L2",
        "description": "Semantic document store and retrieval",
        "distractor_pool": ["web_search", "memory_search", "obsidian"],
    },
    "enterprise-inference": {
        "tier": "T2", "max_data_level": "L2",
        "description": "Managed AI generation via enterprise gateway",
        "distractor_pool": ["openai-whisper", "data-analysis", "web_search"],
    },
    "enterprise-vault": {
        "tier": "T2", "max_data_level": "L4",
        "description": "Encrypted secret store (vault-permission required)",
        "distractor_pool": ["memory_write", "write", "notion"],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 3 — 3P API (L1 only; L2+ strictly prohibited without exception)
    # ─────────────────────────────────────────────────────────────────────────
    "api-gateway": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Unified gateway to 100+ production 3P tools",
        "distractor_pool": ["exec", "enterprise-inference"],
    },
    "web_search": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Public web search",
        "distractor_pool": ["enterprise-rag", "memory_search", "ontology"],
    },
    "web_fetch": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Fetch URL content",
        "distractor_pool": ["nano-pdf", "read", "enterprise-rag"],
    },
    "browser": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Headless browser with web access",
        "distractor_pool": ["enterprise-inference", "nano-pdf"],
    },
    "agent-browser": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Headless browser with unrestricted external access",
        "distractor_pool": ["enterprise-inference", "read"],
    },
    "github": {
        "tier": "T3", "max_data_level": "L1",
        "description": "GitHub REST API",
        "distractor_pool": ["write", "exec", "skill-creator"],
    },
    "slack": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Slack messaging API",
        "distractor_pool": ["enterprise-mail", "write"],
    },
    "notion": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Notion pages and databases",
        "distractor_pool": ["obsidian", "write", "productivity"],
    },
    "gog": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Google Workspace (Docs, Sheets, Drive, Gmail)",
        "distractor_pool": ["word-docx", "excel-xlsx", "enterprise-mail"],
    },
    "trello": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Trello boards and cards",
        "distractor_pool": ["productivity", "cron", "notion"],
    },
    "weather": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Public weather API",
        "distractor_pool": ["web_search"],
    },
    "news-summary": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Public news aggregation",
        "distractor_pool": ["web_search", "enterprise-rag"],
    },
    "stock-analysis": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Public market data and financials",
        "distractor_pool": ["data-analysis", "cfo"],
    },
    "polymarket": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Prediction market odds",
        "distractor_pool": ["web_search", "stock-analysis"],
    },
    "caldav-calendar": {
        "tier": "T3", "max_data_level": "L1",
        "description": "CalDAV sync with external providers",
        "distractor_pool": ["enterprise-calendar", "cron"],
    },
    "clawhub": {
        "tier": "T3", "max_data_level": "L1",
        "description": "ClawHub skill registry",
        "distractor_pool": ["skill-creator", "exec"],
    },
    "academic-research": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Multi-cycle web research with citations",
        "distractor_pool": ["enterprise-rag", "web_search"],
    },
    "flight-search": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Google Flights search",
        "distractor_pool": ["web_search", "goplaces"],
    },
    "goplaces": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Place search, geocoding, routing",
        "distractor_pool": ["web_search", "weather"],
    },
    "eventbrite": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Event management API",
        "distractor_pool": ["enterprise-calendar", "trello"],
    },
    "openai-whisper-api": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Transcription via OpenAI cloud endpoint",
        "distractor_pool": ["openai-whisper", "enterprise-inference"],
    },
    "edge-tts": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Text-to-speech via Microsoft Edge service",
        "distractor_pool": ["enterprise-inference", "openai-whisper"],
    },
    "music-cog": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Generate royalty-free music",
        "distractor_pool": ["enterprise-inference"],
    },
    "moltspaces": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Live audio rooms API",
        "distractor_pool": ["slack", "enterprise-mail"],
    },
    "legaldoc-ai": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Contract analysis and legal research",
        "distractor_pool": ["nano-pdf", "enterprise-rag"],
    },
    "plan2meal": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Recipe and grocery management",
        "distractor_pool": ["productivity", "write"],
    },
    "sudoku": {
        "tier": "T3", "max_data_level": "L1",
        "description": "Fetch and solve Sudoku puzzles",
        "distractor_pool": ["web_fetch"],
    },
}


def get_tool_tier(tool_name: str) -> str:
    """Return the trust tier for a tool, defaulting to T3 (most restrictive) if unknown."""
    entry = TOOL_REGISTRY.get(tool_name)
    if entry:
        return entry["tier"]
    return "T3"


def get_tool_max_level(tool_name: str) -> str:
    """Return the maximum data level a tool can handle by default."""
    entry = TOOL_REGISTRY.get(tool_name)
    if entry:
        return entry["max_data_level"]
    return "L1"


def get_distractors(tool_name: str) -> list[str]:
    """Return distractor tools for COVERT-style augmentation."""
    entry = TOOL_REGISTRY.get(tool_name)
    if entry:
        return entry.get("distractor_pool", [])
    return []


def get_tools_by_tier(tier: str) -> list[str]:
    """Return all tool names for a given tier."""
    return [name for name, info in TOOL_REGISTRY.items() if info["tier"] == tier]


def get_alternative_tool(tool_name: str, target_tier: str) -> Optional[str]:
    """Find a plausible alternative tool in the target tier for the same task category."""
    entry = TOOL_REGISTRY.get(tool_name)
    if not entry:
        return None
    for distractor in entry.get("distractor_pool", []):
        d_entry = TOOL_REGISTRY.get(distractor)
        if d_entry and d_entry["tier"] == target_tier:
            return distractor
    tier_tools = get_tools_by_tier(target_tier)
    return tier_tools[0] if tier_tools else None
