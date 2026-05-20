# Privacy SFT Pipeline

Multi-stage pipeline that converts OpenClaw agent trajectories into privacy-compliant SFT training data and generates RLHF preference pairs for privacy-aware behavior training.

## Pipeline Stages

```
Stage 1: Parse       → Extract trajectory from OpenClaw export zip
Stage 2: Classify    → Multi-engine PII detection (Presidio, DLP, LLM)
Stage 3: Rewrite     → Claude Opus 4.6 injects privacy behaviors per Hatch Trust Guidelines
Stage 4: Verify      → GPT-5.4 scores compliance (auto-refix loop for minor issues)
Stage 5: Write       → Output trajectory.jsonl + workspace + SFT dataset
Stage 6: RLHF       → Generate step-level preference pairs (9 rejected alternatives per decision point)
```

## Hatch Trust Guidelines (HTG) Framework

| Data Level | Examples | Allowed Tiers |
|-----------|----------|---------------|
| L0 Public | General knowledge, broad locations | T1, T2, T3 |
| L1 Caution | Device IDs, IP addresses | T1, T2, T3 |
| L2 Sensitive | Full names, emails, account IDs | T1, T2 only |
| L3 Highly Sensitive | Health records, financials, GPS | T1 only (elicit per-turn) |
| L4 Critical | SSNs, passwords, biometrics | T1 only (zero retention) |

| Tool Tier | Description | Max Data Level |
|-----------|-------------|----------------|
| T1 Local | exec, read, write, memory_search | L4 |
| T2 1P Cloud | enterprise-mail, enterprise-rag | L2 (L3/L4 with consent) |
| T3 3P API | web_search, slack, notion, github | L1 only |

## RLHF Pair Generation

At each privacy-critical decision point in the trajectory, the generator creates 9 suboptimal alternatives:

- **~4 privacy violations** — wrong tool tier, PII leak, skipped consent
- **~3 over-refusals** — refuses legitimate operations unnecessarily
- **~2 hallucinations** — incorrect privacy reasoning

Each alternative is scored 0.0–1.0 by a Process Reward Model (GPT-5.4):
- 0.0–0.1: Catastrophic (data breach)
- 0.3–0.5: Moderate violation
- 0.5–0.7: Over-refusal (annoying but safe)
- 1.0: Optimal (the chosen action)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and OPENAI_API_KEY

# Run pipeline on a single task
python run_pipeline.py --input sample_input.csv --task-id T-033-02

# Run RLHF generation only (dry-run to see decision points)
python test_rlhf.py --dry-run

# Run full RLHF generation with LLM calls
python test_rlhf.py
```

## Comparison UI

A React + FastAPI app for side-by-side viewing of Original SFT, Privacy Rewrite, and RLHF pairs.

```bash
# Start backend
cd comparison-ui/backend
pip install -r requirements.txt
python -m uvicorn server:app --port 8000

# Start frontend (separate terminal)
cd comparison-ui/frontend
npm install
npm run dev
# Open http://localhost:5173
```

## Project Structure

```
privacy-sft-pipeline/
├── run_pipeline.py          # Main CLI orchestrator (Stages 1-6)
├── parser.py                # Stage 1: Parse OpenClaw export zips
├── classifier.py            # Stage 2: Multi-engine PII classification
├── rewriter.py              # Stage 3: Claude-based privacy rewriting
├── verifier.py              # Stage 4: GPT-5.4 verification + refix loop
├── writer.py                # Stage 5: Output writer (JSONL, SFT, RLHF)
├── rlhf_generator.py        # Stage 6: RLHF preference pair generation
├── reward_scorer.py         # Process Reward Model (continuous 0-1 scoring)
├── tool_tiers.py            # Tool tier registry (66 tools → T1/T2/T3)
├── models.py                # Typed dataclasses for all stages
├── config.py                # API keys, model config, quality gates
├── token_tracker.py         # Token usage tracking
├── prompts/
│   ├── rewriter_system.py   # Claude system prompt for privacy rewriting
│   ├── rewriter_turn.py     # Per-turn user prompt for Claude
│   ├── verifier_system.py   # GPT-5.4 verification prompt
│   ├── rlhf_system.py       # RLHF rejected alternative generation prompt
│   ├── rlhf_scorer.py       # Process Reward Model scoring prompt
│   └── classifier_prompt.py # PII classification prompt
├── data/
│   ├── scenarios.json       # Privacy scenario definitions (A-I)
│   ├── classification.json  # PII label → level mapping
│   └── tool_tiers.json      # Tool tier reference data
├── comparison-ui/
│   ├── backend/server.py    # FastAPI serving trajectory data
│   └── frontend/            # React + Vite + Tailwind comparison viewer
└── test_rlhf.py             # Standalone RLHF test script
```

## Output Formats

### SFT Dataset (`sft_dataset.jsonl`)
Standard chat format with user/assistant messages including privacy reasoning in thinking blocks.

### RLHF Pairs (`rlhf/rlhf_pairs.jsonl`)
Step-level preference pairs with chosen + 9 rejected alternatives per decision point.

### DPO Format (`rlhf/rlhf_dpo.jsonl`)
One chosen/rejected pair per line, ready for DPO training frameworks.

## CLI Flags

```
--input, -i       Input CSV (worker_id, submission_id, task_id, export_url)
--output, -o      Output directory (default: ./output)
--task-id, -t     Process single task by ID
--resume, -r      Resume from checkpoint
--dry-run         Parse + classify only (no LLM rewriting)
--concurrency, -c Max concurrent tasks (default: 8)
--no-rlhf         Skip RLHF pair generation (Stage 6)
```
