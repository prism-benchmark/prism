# Unified LLM Configuration Guide

> **Single file. Single client. Run everything or just what you need.**

Everything lives in **`llm_config.yaml`** at the repo root. Every LLM call in the project reads from this one file.

---

## Quick Start

```bash
# 1. Copy env template and fill in your API keys
cp .env.example .env

# 2. Edit llm_config.yaml to enable/disable what you want
# 3. Run
python run.py                       # everything enabled
python run.py --profile quick       # smoke test
python run.py --only constructiveness,novelty   # specific aspects
python run.py --list                # see what's available
```

---

## Config Structure (`llm_config.yaml`)

```
llm_config.yaml
├── run_profiles          ← named subsets for quick selection
├── providers             ← API keys & endpoints (one per provider)
├── defaults              ← global fallback model/params
├── provider_defaults     ← per-provider fallback model/params
├── aspects               ← benchmarking evaluators (enabled flag + model)
│   ├── constructiveness
│   ├── depth_of_analysis
│   ├── flaw_identification
│   │   ├── step1
│   │   └── step2
│   └── novelty
└── reviewers             ← LLM reviewers (enabled flag + type)
    ├── sea               (local / vLLM)
    ├── reviewer2         (local / vLLM)
    ├── treereview        (API)
    ├── deepreview        (local / vLLM)
    └── cyclereview       (local / vLLM)
```

---

## Enable / Disable Components

Set `enabled: false` on any aspect or reviewer to skip it entirely:

```yaml
aspects:
  novelty:
    enabled: false          # ← won't run, won't load config

  constructiveness:
    enabled: true           # ← active (default if key omitted)
    provider: mimo
    model: openai/gpt-oss-20b

reviewers:
  deepreview:
    enabled: false          # ← skip this reviewer

  treereview:
    enabled: true
    type: api
    provider: gemini
```

**Rules:**
- If `enabled` is omitted → defaults to `true`
- Disabled items are invisible to `list_enabled()` and `resolve_items()`
- You can still force-run a disabled item with `--only`

---

## Run Profiles

Profiles are named subsets defined in `llm_config.yaml`:

```yaml
run_profiles:
  all: null                 # no filter — uses enabled flags only
  aspects:
    - constructiveness
    - depth_of_analysis
    - flaw_identification
    - novelty
  reviewers:
    - sea
    - reviewer2
    - treereview
    - deepreview
    - cyclereview
  quick:                    # fast smoke test
    - constructiveness
    - treereview
  heavy:                    # full evaluation
    - constructiveness
    - depth_of_analysis
    - flaw_identification
    - novelty
    - treereview
    - deepreview
```

**Filtering logic:**
- `null` → no filter (all enabled items run)
- List of names → only those items run (still respects `enabled` flag)

### Add your own profile

```yaml
run_profiles:
  my_experiment:
    - constructiveness
    - novelty
    - treereview
```

```bash
python run.py --profile my_experiment
```

---

## CLI Usage

### Run everything enabled

```bash
python run.py
```

### Run a profile

```bash
python run.py --profile quick
python run.py --profile aspects
python run.py --profile reviewers
python run.py --profile heavy
```

### Run specific items (whitelist)

```bash
# Only these aspects
python run.py --only constructiveness,novelty

# Only these reviewers
python run.py --only treereview,sea

# Mix aspects and reviewers
python run.py --only constructiveness,treereview,novelty
```

### Skip specific items (blacklist)

```bash
# Run everything except these
python run.py --skip deepreview,cyclereview

# Combine with profile
python run.py --profile heavy --skip flaw_identification
```

### List available items

```bash
python run.py --list
```

Output:
```
ASPECTS:
  ✓ constructiveness    (mimo / openai/gpt-oss-20b)
  ✓ depth_of_analysis   (mimo / openai/gpt-oss-20b)
  ✓ flaw_identification (mimo / openai/gpt-oss-20b)  [2 steps]
  ✗ novelty             DISABLED

REVIEWERS:
  ✓ sea                 (local: Qwen/Qwen2-7B-Instruct)
  ✓ reviewer2           (local: Qwen3-14B)
  ✓ treereview          (api: gemini / gemini-2.0-flash)
  ✓ deepreview          (local: 14B)
  ✓ cyclereview         (local: 8B)

PROFILES: all, aspects, reviewers, quick, heavy
```

### Combine filters

```bash
# Profile + skip
python run.py --profile aspects --skip depth_of_analysis

# Profile + only (only takes priority)
python run.py --profile heavy --only constructiveness
```

---

## Python API

### Using `PRISMLLMClient` (recommended for new code)

```python
from llm_client import PRISMLLMClient

# Client for a benchmarking aspect
client = PRISMLLMClient.for_aspect("constructiveness")
result = client.generate_text("You are a judge.", "Score this review...")

# Multi-step aspect
client = PRISMLLMClient.for_aspect("flaw_identification", step="step1")

# Client for an API-based reviewer
client = PRISMLLMClient.for_reviewer("treereview")

# Override specific settings
client = PRISMLLMClient.for_aspect("novelty", model="gpt-4o", temperature=0.1)
```

### Listing and filtering

```python
from llm_client import (
    list_enabled, list_disabled, list_all,
    is_enabled, resolve_items,
    get_profile_names, get_profile_items,
)

# What's available
list_all("aspects")         # ['constructiveness', 'depth_of_analysis', 'flaw_identification', 'novelty']
list_enabled("aspects")     # only enabled ones
list_disabled("aspects")    # only disabled ones

# Check a specific item
is_enabled("aspects", "novelty")  # True or False

# Resolve with filters (for your run loop)
items = resolve_items("aspects", profile="quick")
# → ['constructiveness']  (only constructiveness is in "quick" profile AND enabled)

items = resolve_items("aspects", only=["novelty", "constructiveness"])
# → ['constructiveness', 'novelty']  (--only overrides enabled flag)

items = resolve_items("reviewers", skip=["deepreview"])
# → enabled reviewers minus deepreview

# Profiles
get_profile_names()                # ['all', 'aspects', 'heavy', 'quick', 'reviewers']
get_profile_items("quick")         # ['constructiveness', 'treereview']
get_profile_items("all")           # None (no filter)
```

### Legacy interface (still works)

```python
from ai_config import get_llm_client, get_provider_config, validate_env

# Same factory — routes to aspects or reviewers automatically
client = get_llm_client("constructiveness")
client = get_llm_client("treereview")
client = get_llm_client("flaw_identification", step="step1")

# Provider config dict (for legacy adapters)
cfg = get_provider_config("mimo")
# → {'api_key': '...', 'base_url': '...', 'default_model': 'mimo-v2.5-pro', ...}

# Validate required env vars
validate_env()
```

---

## Provider Setup

Each provider needs credentials in `llm_config.yaml` + matching env vars in `.env`:

```yaml
# llm_config.yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}          # from .env
    base_url: https://api.openai.com/v1

  mimo:
    api_key: ${MIMO_API_KEY}
    base_url: ${MIMO_BASE_URL:https://api.xiaomimimo.com/v1}
    #                                 ↑ default if env var not set

  devmate:
    api_key: ${DEVMATE_API_KEY}
    base_url: ${DEVMATE_BASE_URL:}
    proxy: ${DEVMATE_PROXY:}
    disable_ssl_verify: ${DEVMATE_DISABLE_SSL_VERIFY:true}
```

```bash
# .env
OPENAI_API_KEY=sk-...
MIMO_API_KEY=...
GOOGLE_API_KEY=...
DEVMATE_API_KEY=...
LLM_API_KEY=...           # used by openrouter
```

### Built-in providers

| Provider      | SDK           | Notes                               |
|---------------|---------------|-------------------------------------|
| `openai`      | openai        | Native OpenAI API                   |
| `gemini`      | google-genai  | Native Gemini SDK                   |
| `azure`       | openai        | Azure OpenAI (needs deployment)     |
| `mimo`        | openai        | Xiaomi Mimo (OpenAI-compatible)     |
| `devmate`     | openai        | Custom Devmate (with proxy/SSL)      |
| `openrouter`  | openai        | OpenRouter (OpenAI-compatible)      |

### Adding any OpenAI-compatible provider

No code changes needed. Add a new entry under `providers` and reference it:

```yaml
# llm_config.yaml
providers:
  together:
    api_key: ${TOGETHER_API_KEY}
    base_url: https://api.together.xyz/v1

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    base_url: https://api.anthropic.com/v1

aspects:
  constructiveness:
    provider: together
    model: meta-llama/Llama-3.3-70B-Instruct-Turbo
    temperature: 0.0
```

```bash
# .env
TOGETHER_API_KEY=tsk_...
ANTHROPIC_API_KEY=sk-ant-...
```

The system automatically resolves `{PROVIDER}_API_KEY` and `{PROVIDER}_BASE_URL` from `.env`. Unknown providers are treated as OpenAI-compatible.

---

## Data Setup

```bash
# Download and prepare the full benchmark dataset:
python run.py --setup-data

# Or manually:
python Data/setup_aspect_benchmark.py --write-env
```

---

## How Filtering Priority Works

When multiple filters are active, they combine like this:

```
                        ┌─────────────────┐
                        │   --only list   │  ← HIGHEST PRIORITY
                        │  (whitelist)    │     overrides everything
                        └────────┬────────┘
                                 │ if --only is empty:
                        ┌────────▼────────┐
                        │  --profile items │  ← intersect with enabled
                        │  (profile list)  │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  enabled flags   │  ← base set
                        │  (from YAML)     │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │   --skip list   │  ← ALWAYS applied last
                        │  (blacklist)     │     removes from result
                        └─────────────────┘
```

**Examples:**

| Scenario | Result |
|----------|--------|
| No flags | All enabled items |
| `--profile quick` | Items in "quick" ∩ enabled |
| `--only novelty` | Just "novelty" (even if disabled) |
| `--only novelty --skip novelty` | Empty (skip always applies) |
| `--profile aspects --skip novelty` | Aspects in profile ∩ enabled, minus novelty |
| `--only a,b --profile quick` | Just [a, b] (--only overrides profile) |

---

## Common Recipes

### Full paper experiment (all 4 aspects, all conferences, all reviewers)

```bash
python run.py
```

### Dry-run to preview what would execute

```bash
python run.py --dry-run
```

### Run only benchmarking aspects (no LLM reviewers)

```bash
python run.py --profile aspects
# or
python run.py --only constructiveness,depth_of_analysis,flaw_identification,novelty
```

### Run only LLM reviewers (no aspects)

```bash
python run.py --profile reviewers
```

### Quick smoke test before full run

```bash
python run.py --profile quick
```

### Test a single aspect with a specific model

```bash
python run.py --only constructiveness
```

Or in Python:

```python
client = PRISMLLMClient.for_aspect("constructiveness", model="gpt-4o")
```

### Run on specific conferences

```bash
python run.py --conference iclr2024,neurips2025
```

### Process a subset of papers (for quick iteration)

```bash
python run.py --only constructiveness --limit 10
```

### Forward extra arguments to underlying pipeline

```bash
python run.py --only constructiveness -- --workers 4 --with-paper
```

### Setup / download the benchmark dataset

```bash
python run.py --setup-data
```

### Disable slow reviewers temporarily

Edit `llm_config.yaml`:
```yaml
deepreview:
  enabled: false
cyclereview:
  enabled: false
```

Then `python run.py` skips them automatically.

### Add a new aspect

```yaml
aspects:
  my_new_aspect:
    enabled: true
    provider: mimo
    model: openai/gpt-oss-20b
    temperature: 0.0
    max_tokens: 4000
```

Then use: `PRISMLLMClient.for_aspect("my_new_aspect")`

### Add a new API reviewer

```yaml
reviewers:
  my_reviewer:
    enabled: true
    type: api
    provider: openai
    model: gpt-4o
    temperature: 0.0
    max_tokens: 8192
```

Then use: `PRISMLLMClient.for_reviewer("my_reviewer")`

### Switch all aspects to a different provider

```yaml
defaults:
  provider: openai          # change global default
  model: gpt-4o-mini        # change global model

# Or per-aspect:
aspects:
  constructiveness:
    provider: openai        # override just this one
    model: gpt-4o
```

### Validate env before running

```python
from ai_config import validate_env

validate_env()  # raises EnvironmentError if API keys missing
```

---

## File Map

```
PRISM/
├── run.py                                       ← Unified experiment orchestrator
├── llm_config.yaml                              ← SINGLE SOURCE OF TRUTH
├── llm_client.py                                ← Config loader + PRISMLLMClient
├── ai_config.py                                 ← Thin facade (legacy re-exports)
├── .env                                         ← API keys (gitignored)
├── tests/
│   └── test_llm_config.py                       ← 22 tests
├── Aspects_benchmarking/
│   ├── env_loader.py                            ← Path helpers only
│   ├── constructiveness/src/evaluator.py        ← Uses get_llm_client("constructiveness")
│   ├── depth_of_analysis/config.py              ← Imports from ai_config
│   ├── flaw_identification/src/evaluator.py     ← Uses get_llm_client("flaw_identification")
│   └── novelty_vefification/
│       ├── config.py                            ← Imports from ai_config
│       └── services/llm_client.py               ← Legacy client (to be migrated)
└── LLM_reviewer/
    ├── SEA/vllm_config.py                       ← Uses get_reviewer_config("sea")
    ├── TreeReview/.../LLMClient.py              ← Uses ai_config
    └── Deepreview_CycleReview/run/              ← Uses get_reviewer_config()
```

---

## Migration Notes

| Old pattern | New equivalent |
|-------------|----------------|
| `from ai_config import get_llm_client` | Same — still works |
| `from ai_config import get_provider_config` | Same — still works |
| `from ai_config import GOOGLE_API_KEY` | Same — still works |
| `from env_loader import ...` | Same — still works |
| Hardcoded `config.py` per submodule | Just import from `ai_config` |

**For new code, prefer:**
```python
from llm_client import PRISMLLMClient, resolve_items, list_enabled
```

**For backward compat, keep:**
```python
from ai_config import get_llm_client, get_provider_config, GOOGLE_API_KEY
```
