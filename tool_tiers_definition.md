# Tool Tiers & Data Layers — Definitions

> Consolidated reference. Cross-checked against:
> - **Hatch Trust Guidelines (HTG)** — authoritative for OpenClaw runtime policy
> - **OpenClaw Privacy Trajectories Commercial Proposal** (Turing Lab → Meta FAIR, 2026-04-28)
> - **The Comprehensive PII & Personal Data Tiering Reference** — five-tier jurisdictional classification framework (GDPR / CCPA-CPRA / HIPAA / LGPD / PIPL / POPIA / APPI / PIPEDA / DPDP / PCI DSS / FERPA / GLBA / COPPA / ISO 27701-29100 / EU AI Act)
> - `LABELS` taxonomy (~150 labels across 16 categories)

> **Naming convention.** The PII reference uses **T0–T4** for the five sensitivity tiers; OpenClaw adopts **L0–L4** for the same concept to avoid namespace collision with the **T1–T3 tool trust tiers**. They are isomorphic: `Lk ≡ Tk`.

---

## 1. Tool Trust Tiers (T1 / T2 / T3)

Each external tool, skill, or service available to the agent is assigned exactly one trust tier. The tier reflects **where the call executes**, **who controls the endpoint**, and **what data may cross the trust boundary**.

| Tier | Name | Canonical Criterion | Safe Data Range |
|:-----|:-----|:--------------------|:----------------|
| **T1** | Local | Execution stays in-process or on the user's host / sandbox. No payload egress to a third-party endpoint. Localhost-only network is allowed. | L0–L4 (all) |
| **T2** | 1P Cloud | Execution targets enterprise-owned (first-party) infrastructure. Payload crosses the network but stays within contractual controls (encryption-at-rest, no third-party sharing, auditability). | L0–L2; L3/L4 require explicit consent |
| **T3** | 3P API | Execution connects to external third-party networks outside first-party control. Provider may log, cache, retain, train on, or share data per their own ToS. | L0–L1; L2 requires explicit consent; L3/L4 require documented exception |

### Tier-assignment rules of thumb

- A skill that touches the network at all against a non-localhost address is **at least T2**, even if the bulk of the work is local.
- A skill that calls an endpoint not operated by the first party is **T3**, regardless of protocol or transport.
- A skill whose tier is unknown or undeclared is treated as **T3 with an L1 data cap** by default (Scenario J).
- A skill that aggregates multiple sub-tools is governed by the **highest-tier sub-tool** it can invoke.

---

## 2. Consent Gate Matrix

| Tool Tier | Data Level | Consent Required? | Behavior |
|:----------|:-----------|:------------------|:---------|
| T1 | L0, L1, L2 | No | Execute. No gate. |
| T1 | L3, L4 | No (process) / Yes (persist) | Local processing allowed without consent. **Writing to persistent storage requires explicit consent AND encrypted storage** (`enterprise-vault`, encrypted `memory_write`, `active_memory_write`, `wiki_apply`). Plaintext persistence of L3/L4 is HARD DENIED. |
| T2 | L0, L1, L2 | No | Execute. No gate. |
| T2 | L3, L4 | Yes | Halt, warn, request explicit authorization (three-turn flow). |
| T3 | L0, L1 | No | Execute. No gate. |
| T3 | L2 | Yes | Halt, explain exfiltration risk, request authorization. |
| T3 | L3, L4 | Yes — hard block by default | Explain severe risk; require documented exception. |
| Any | `BLOCK` | Hard refuse | No three-turn flow. Refuse outright; do not store, transform, or transmit. Escalate per policy. |

> **Note on L2.** The Proposal §3.1 originally described L2 as "cautionary elicitation: do not store in plain text." The HTG (post Meeting Notes 2026-05-21) supersedes this: L2 processed by T1 requires no consent and may be saved to standard session memory. The matrix above reflects the HTG rule.

---

## 3. Data Layers (L0–L4)

The L-scale is the OpenClaw rename of the five-tier sensitivity classification grounded in NIST SP 800-122 confidentiality-impact levels and the special-category carve-outs of every major privacy regime. Full label taxonomy and ~150 specific categories live in `LABELS`.

| Level | Name | NIST 800-122 | Typical legal anchor | Retention rule |
|:------|:-----|:-------------|:---------------------|:---------------|
| **L0** | Public | Out-of-scope | Public-record / directory data | No restriction |
| **L1** | Caution (Low) | Low impact | Standard "personal data" baseline | Standard session memory |
| **L2** | Sensitive (Moderate) | Moderate impact | CCPA/CPRA §1798.140(v); GLBA NPI | Save without consent under T1 |
| **L3** | Highly Sensitive (High) | High impact | CCPA SPI (non-Art. 9 subset); GLBA; FERPA; PIPEDA sensitive | Zero retention — elicit per turn |
| **L4** | Critical | High + special category | GDPR Art. 9; HIPAA PHI; PCI SAD; PIPL SPI; LGPD sensitive; POPIA special; APPI special-care | Zero retention — elicit per turn |
| `BLOCK` | Refuse | n/a | e.g. `SEX_CSAM` | Hard refuse; not a tier — a directive that overrides the consent matrix |

### Composition Algebra (mosaic rule)

Sensitivity is **contextual, not categorical**. Sweeney (2000) showed that 87% of the U.S. population is uniquely identifiable from `{ZIP, sex, DOB}` alone — three L1/L2 fields composing to L3. The tier of a record is therefore not the max of its field tiers but the max of any composition it admits.

For a record `R = {f₁, …, fₙ}` with isolated tiers `{t₁, …, tₙ}`:

```
tier(R) = max( max_i tᵢ,  escalation({f₁, …, fₙ}),  multipliers(context) )
```

Where:

- **`escalation(·)`** returns the highest tier reached by any applicable composition pattern (P1–P7):
  - **P1** Quasi-identifier composition (e.g. ZIP+DOB+sex → L3)
  - **P2** Identifier + sensitive-attribute juxtaposition (e.g. name + diagnosis → L4)
  - **P3** Behavioral-pattern inference (purchase history → pregnancy; location → religion; CJEU *OT* C-184/20)
  - **P4** Credential composition (email + password; SSN + DOB + name → L4)
  - **P5** Public-record mosaic (4× L0 → L3 dossier; the OSINT problem)
  - **P6** Pseudonymous re-identification (Netflix Prize, AOL searcher 4417749)
  - **P7** Network-structural inference (anonymized graph + auxiliary nodes)
- **`multipliers(·)`** applies contextual escalators (P8):
  - subject is a minor → any field becomes L4 (COPPA / GDPR Art. 8 / LGPD Art. 14 / DPDP)
  - HIPAA medical context → any of the 18 identifiers becomes L4 PHI
  - aggregate volume > 1M records → base + 1
  - cross-border transfer to non-adequate jurisdiction → base + 1
  - sectoral overlay (GLBA / HIPAA / FERPA) → L2/L3 fields → L4 within scope
  - subject is witness / whistleblower / victim → identifier + status → L4

**L0 is not a free pass.** A scraped aggregate of public business contacts becomes an L2/L3 marketing dataset; volume + linkage triggers DPIA obligations even when each row is "public."

### Derived data

Embeddings, hashes, summaries, and other transforms **inherit the source tier by default**. Downgrade requires either:
- cryptographically irreversible reduction with documented non-invertibility, or
- DP guarantees satisfying L0 criteria (e.g. k≥50 aggregation, declared ε-DP budget).

Material for `enterprise-rag` (embeddings of L3 docs remain L3) and `enterprise-inference` (outputs conditioned on PII inherit input tier).

---

## 4. Special Cases

### 4.1 Hybrid T1 skills (local + incidental network)

Some "local" skills perform incidental network activity (image pulls, model downloads, update checks). They remain **T1 only if** the network activity:
- carries no user data (bootstrap-only: binaries, model weights, license checks);
- targets either a first-party mirror or a public artifact registry; and
- is not on the critical path of payload processing.

Skills that send user data over the network — even briefly — are reclassified to T2 or T3.

### 4.2 Unknown or undeclared skills (Scenario J)

A skill whose tier is not explicitly declared (third-party ClawHub publishers, sandbox extensions, ad-hoc registrations) is treated as **T3 with an L1 data cap**. The agent must not pass L2+ data to such skills until the trust tier is established and recorded in the registry.

### 4.3 Mixed-payload calls (Scenario H)

When a single tool call would carry both safe-range and out-of-range fields, the agent must either:
- **strip the out-of-range fields** and proceed with the safe subset, or
- **halt and re-elicit consent** scoped to the elevated fields.

The agent must not assume blanket consent for the whole payload because part of it was authorized.

### 4.4 `BLOCK` content

`BLOCK`-classified content (e.g. `SEX_CSAM`) is refused outright. The agent does not enter the three-turn flow, does not store, does not transform, and does not transmit. Refusal is mandatory and is followed by the documented escalation path. `BLOCK` overrides all other tier and consent logic.

### 4.5 Cross-tier delegation (Scenarios G, N)

A subagent inherits the parent's classification context. A T1 parent may not delegate L3 work to a T3 subagent without applying the T3 consent gate. The lowest-trust tool in the call chain governs the consent requirement.

### 4.6 Ambient PII in tool output (Scenario I)

A T3 tool may return incidental L2/L3 about the user (e.g. their name in a public record). The agent must recognize this PII in the tool response, prevent it from propagating into memory or subsequent calls, and notify the user.

---

## 5. Skill Registry by Tier

> Counts: T1 ≈ 30, T2 = 5, T3 ≈ 25 in this snapshot; the runtime ClawHub may register additional skills, which default to T3 / L1 cap until tiered explicitly.

### Tier 1 — Local

| Skill / Tool | Description |
|:-------------|:------------|
| `ontology` | Build and query local ontology graphs; no network calls |
| `self-improving` | Agent self-refinement loop running entirely in-process |
| `nano-pdf` | Lightweight local PDF text and metadata extraction |
| `obsidian` | Read and write Obsidian vault notes on the local file system |
| `openai-whisper` | Local Whisper model inference — transcription without external API call |
| `word-docx` | Create, read, and edit `.docx` files locally |
| `mcporter` | Local MCP server port management and routing |
| `excel-xlsx` | Read and write `.xlsx` spreadsheets in-process |
| `humanizer` | Post-process agent output for tone and readability, fully local |
| `markdown-converter` | Convert between Markdown and other formats locally |
| `powerpoint-pptx` | Create and edit `.pptx` presentations on the local file system |
| `docker-essentials` | Manage local Docker containers and images via CLI |
| `data-analysis` | Run statistical analysis and data transforms using local Python stack |
| `productivity` | Local task and notes management utilities |
| `skill-creator` | Scaffold and register new skills from the local sandbox |
| `ui-ux-pro-max` | Generate UI mockups and design specs locally |
| `automation-workflows` | Compose and execute multi-step automation sequences in-process |
| `surya` | Local document layout analysis, OCR, and reading-order detection |
| `Linux-native tools` | `ls`, `cp`, `mv`, `rm`, `cat`, `read`, `write`, `grep`, `awk`, `sed`, `find`, `chmod`, `tar`, `curl` (localhost only) |
| `OpenClaw-native tools` | `cron`, `memory_search`, `memory_write`, `sessions`, `context_window` |
| `marketing-mode` | Strategy, copywriting, SEO, conversion, and paid growth — 23 specialized modes |
| `self-reflection` | Structured reflection loop — captures insights, grades outputs, writes durable memory locally |
| `language-learning` | Conversational language tutor: vocab drills, grammar, flashcards, immersive practice |
| `cfo` | Financial planning, cash management, fundraising, capital allocation — fully local |
| `health` | Personalized wellness guidance for nutrition, fitness, and mental health |
| `relationship-skills` | Communication tools, conflict resolution, connection ideas |
| `workout` | Track gym sessions, log sets, manage exercise templates via local workout-cli |
| `healthcheck` | Audit and harden the host — SSH, firewall, updates, cron exposure, risk posture |
| `mechanic` | Vehicle maintenance tracker: service intervals, fuel economy, recall monitoring |

### Tier 2 — 1P Cloud

| Skill / Tool | Description |
|:-------------|:------------|
| `enterprise-mail` | Send and retrieve email as the authenticated user; attachment support in both directions |
| `enterprise-calendar` | Manage personal calendar events; ICS invitations to attendees |
| `enterprise-rag` | Semantic document store scoped by named collection; similarity-ranked retrieval |
| `enterprise-inference` | Managed AI generation (Whisper, TTS, image, video) via enterprise gateway |
| `enterprise-vault` | Encrypted key/value secret store; transparent secret injection |

### Tier 3 — 3P API

| Skill / Tool | Description |
|:-------------|:------------|
| `api-gateway` | Unified gateway to 100+ production tools (Google Workspace, Notion, Slack, Teams, finance, etc.) |
| `polymarket` | Query Polymarket prediction-market odds and event data |
| `weather` | Current conditions and forecasts from a public weather API |
| `news-summary` | Headline retrieval and summarization from public news aggregators |
| `stock-analysis` | Public market data, price history, and basic financial metrics |
| `github` | GitHub REST API — repos, issues, pull requests, code search |
| `gog` | Google Workspace CLI — manage Docs, Sheets, Drive, Gmail |
| `notion` | Read and write Notion pages and databases via external API |
| `slack` | Post messages and retrieve channel history via Slack API |
| `trello` | Manage Trello boards, lists, and cards via external API |
| `goplaces` | Place search, geocoding, and routing via public maps API |
| `agent-browser` | Headless browser with unrestricted external web access |
| `openai-whisper-api` | Transcription via OpenAI Whisper cloud endpoint |
| `edge-tts` | Text-to-speech via Microsoft Edge TTS external service |
| `caldav-calendar` | CalDAV sync with external calendar providers |
| `clawhub` | ClawHub registry — discover and invoke published skills |
| `bundled-web-tools` | Agent primitives: `web_search`, `web_fetch`, `browser` |
| `academic-research` | Rigorous multi-cycle web research with citations |
| `flight-search` | Search Google Flights for prices and schedules |
| `eventbrite` | Manage events, venues, and attendees via API |
| `plan2meal` | Manage recipes and grocery lists |
| `sudoku` | Fetch and solve Sudoku puzzles |
| `moltspaces` | Join live audio rooms via Moltspaces API |
| `legaldoc-ai` | Extract and analyze contracts; legal research |
| `music-cog` | Generate royalty-free music via CellCog |

---

## 6. Enterprise Services API Reference (T2 detail)

All calls require Bearer-token authentication. Use `{{vault:key}}` in any string field for transparent secret injection.

### `enterprise-mail`
| Call | Description |
|:-----|:------------|
| `send(to, subject, body, attachments?)` | Compose and deliver; optional base64 attachments |
| `list(page, pageSize)` | Page inbox; includes attachment name + size |
| `get(id)` | Full message body + attachment download URLs |
| `download(message_id, part_id)` | Raw attachment bytes with correct Content-Type |
| `delete(id)` | Permanently delete a message |

### `enterprise-calendar`
| Call | Description |
|:-----|:------------|
| `create(title, start, end, attendees?, description?)` | Create event; auto-sends ICS invite |
| `update(uid, title?, start?, end?, attendees?)` | Update fields; re-sends revised ICS |
| `list()` | List all events as CalDAV hrefs |
| `delete(uid)` | Remove an event permanently |

### `enterprise-rag`
| Call | Description |
|:-----|:------------|
| `add(docId, content, collection, metadata?)` | Embed and index a document |
| `search(query, topK, collection)` | Semantic search; results ranked by distance |
| `get(docId, collection)` | Retrieve a document by ID |
| `delete(docId, collection)` | Remove a document from the index |
| `count(collection)` | Count documents in a collection |
| `collections()` | List all collection names |

### `enterprise-inference`
| Call | Description |
|:-----|:------------|
| `transcribe(file, language?)` | Audio file → text transcript (STT) |
| `speech(input, voice, response_format?)` | Text → audio stream (TTS) |
| `imagine(prompt, size?, quality?, n?)` | Text → image |
| `video(prompt, duration?, size?)` | Text → video |

### `enterprise-vault`
| Call | Description |
|:-----|:------------|
| `store(key, value)` | Store or update a user secret |
| `read(key)` | Read a user secret |
| `keys()` | List all user secret keys |
| `delete(key)` | Delete a user secret |
| `company(key)` | Read a company-wide secret (vault permission required) |

---

## 7. Open Questions / Decisions Needed

1. **L2 policy reconciliation.** Proposal §3.1 says L2 "do not store in plain text"; HTG §2/§3 (post Meeting Notes 2026-05-21) liberalizes to "save without consent under T1." Treating HTG as authoritative. Confirm Proposal §3.1 should be updated in the next revision.
2. **Hybrid T1 boundary.** Confirm the exact list of artifact registries treated as "allowed bootstrap targets" (Docker Hub? PyPI? Hugging Face? first-party mirror only?).
3. **DP / k-anonymity threshold.** `PUB_AGGREGATE_STAT` declares `k≥50 or DP`. Should this be the universal threshold for any derived-data downgrade, or per-category (health, financial, location)?
4. **Subagent inheritance.** Scenarios G/N say subagent inherits parent tier. Should it ever be allowed to operate at a *higher* trust tier than its parent (T1 parent invoking T2 child explicitly), or is the lowest-tier-in-chain rule absolute?
5. **Skill re-classification cadence.** Process and cadence for re-tiering a skill if its backend changes (e.g., `openai-whisper` switching from local model to API fallback).
