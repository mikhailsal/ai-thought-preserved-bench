# ─────────────────────────────────────────────────────────────────────────────
# AI Thought Preservation Bench — Makefile
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   make              Show this help
#   make help         Show this help
#   make run          Run the benchmark (all active models)
#   make test         Run the test suite with coverage
#
# Most benchmark targets accept optional variables:
#   MODELS=…     Comma-separated model config labels or model IDs
#   SCENARIOS=…  Comma-separated scenario IDs (plain_chat_history, tool_mediated_reply)
#   REPS=…       Number of repetitions per model/scenario (default: 5)
#   JUDGE=…      Judge model for evaluation (default: google/gemini-3-flash-preview)
#   WORKERS=…    Max parallel workers (default: 6)
#
# Examples:
#   make run MODELS="x-ai/grok-4.1-fast"
#   make run MODELS="x-ai/grok-4.1-fast" SCENARIOS="plain_chat_history" REPS=3
#   make rejudge MODELS="openai/gpt-oss-20b"
#   make probe MODELS="deepseek/deepseek-v4-flash"
#   make test-file FILE=tests/test_runner_cli_probe.py
#
# ─────────────────────────────────────────────────────────────────────────────

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ─── Configuration ──────────────────────────────────────────────────────────

PYTHON    ?= python
PIP       ?= pip
PYTEST    ?= pytest
VENV_DIR  ?= .venv
CLI       := thought-preserved-bench

# Passthrough variables for benchmark commands (leave empty for defaults)
MODELS    ?=
SCENARIOS ?=
REPS      ?=
JUDGE     ?=
WORKERS   ?=
FILE      ?=

# Inline Python script for 'make models' (define block handles multiline)
define MODELS_SCRIPT
from src.config import get_active_model_configs
configs = get_active_model_configs()
print()
print(f"  Active models: {len(configs)}")
print()
for i, c in enumerate(configs, 1):
    provider = c.provider or "(none)"
    rtype = c.reasoning_type or "?"
    print(f"  {i:>2}. {c.label:<48s} provider={provider:<20s} reasoning={rtype}")
print()
endef
export MODELS_SCRIPT

# Build CLI flag strings only when variables are set
_models    = $(if $(MODELS),--models "$(MODELS)")
_scenarios = $(if $(SCENARIOS),--scenarios "$(SCENARIOS)")
_reps      = $(if $(REPS),--reps $(REPS))
_judge     = $(if $(JUDGE),--judge "$(JUDGE)")
_workers   = $(if $(WORKERS),--workers $(WORKERS))

# ─── Colors & Formatting ───────────────────────────────────────────────────

BOLD    := \033[1m
DIM     := \033[2m
CYAN    := \033[36m
GREEN   := \033[32m
YELLOW  := \033[33m
MAGENTA := \033[35m
BLUE    := \033[34m
RED     := \033[31m
RESET   := \033[0m

# ─── Help ───────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo -e "  $(BOLD)AI Thought Preservation Bench$(RESET)"
	@echo -e "  $(DIM)Benchmark for reasoning replay continuity across normal chat and tool-mediated turns$(RESET)"
	@echo ""
	@echo -e "  $(DIM)Usage:  make <target> [VARIABLE=value …]$(RESET)"
	@echo ""
	@echo -e "  $(BOLD)$(CYAN)Setup$(RESET)"
	@echo -e "    $(GREEN)make install$(RESET)         Install project + test deps in editable mode"
	@echo -e "    $(GREEN)make venv$(RESET)            Create virtualenv, install everything, copy .env"
	@echo -e "    $(GREEN)make env$(RESET)             Copy .env.example → .env (if .env missing)"
	@echo ""
	@echo -e "  $(BOLD)$(CYAN)Benchmark$(RESET)  $(DIM)— all accept MODELS, SCENARIOS, REPS, JUDGE, WORKERS$(RESET)"
	@echo -e "    $(GREEN)make run$(RESET)             Run benchmark (skips cached runs)"
	@echo -e "    $(GREEN)make rerun$(RESET)           Re-run benchmark ignoring all cache"
	@echo -e "    $(GREEN)make run-sequential$(RESET)  Run benchmark without parallelism"
	@echo -e "    $(GREEN)make report$(RESET)          Regenerate reports from cached results"
	@echo -e "    $(GREEN)make rejudge$(RESET)         Re-evaluate cached runs with the judge (cheap)"
	@echo -e "    $(GREEN)make probe$(RESET)           Probe models for reasoning format"
	@echo -e "    $(GREEN)make probe-force$(RESET)     Probe models, ignoring cached probes"
	@echo ""
	@echo -e "  $(BOLD)$(CYAN)Testing$(RESET)"
	@echo -e "    $(GREEN)make test$(RESET)            Run full test suite with coverage"
	@echo -e "    $(GREEN)make test-quick$(RESET)      Run tests without coverage"
	@echo -e "    $(GREEN)make test-file$(RESET)       Run a single test file  $(DIM)FILE=tests/test_foo.py$(RESET)"
	@echo -e "    $(GREEN)make test-match$(RESET)      Run tests matching a pattern  $(DIM)K=test_something$(RESET)"
	@echo -e "    $(GREEN)make coverage$(RESET)        Generate HTML coverage report"
	@echo ""
	@echo -e "  $(BOLD)$(CYAN)Quality$(RESET)"
	@echo -e "    $(GREEN)make check$(RESET)           Run all checks (tests + lint)"
	@echo -e "    $(GREEN)make lint$(RESET)            Run ruff linter"
	@echo -e "    $(GREEN)make format$(RESET)          Auto-format with ruff"
	@echo -e "    $(GREEN)make typecheck$(RESET)       Run mypy type checking"
	@echo ""
	@echo -e "  $(BOLD)$(CYAN)Logs & Info$(RESET)"
	@echo -e "    $(GREEN)make logs$(RESET)            List recent log files"
	@echo -e "    $(GREEN)make logs-tail$(RESET)       Show last 50 lines of latest log"
	@echo -e "    $(GREEN)make models$(RESET)          List all active models from registry"
	@echo -e "    $(GREEN)make leaderboard$(RESET)     Display the current leaderboard"
	@echo -e "    $(GREEN)make cli-help$(RESET)        Show the full CLI help"
	@echo ""
	@echo -e "  $(BOLD)$(CYAN)Maintenance$(RESET)"
	@echo -e "    $(GREEN)make clean$(RESET)           Remove build artifacts and caches"
	@echo -e "    $(GREEN)make clean-logs$(RESET)      Remove all log files"
	@echo -e "    $(GREEN)make clean-probes$(RESET)    Remove cached probe results"
	@echo -e "    $(GREEN)make clean-all$(RESET)       Remove everything (artifacts + logs + probes)"
	@echo ""
	@echo -e "  $(BOLD)$(CYAN)Variables$(RESET)"
	@echo -e "    $(YELLOW)MODELS$(RESET)=$(DIM)\"vendor/model-id\"$(RESET)              Filter by model(s)"
	@echo -e "    $(YELLOW)SCENARIOS$(RESET)=$(DIM)\"plain_chat_history\"$(RESET)        Filter by scenario(s)"
	@echo -e "    $(YELLOW)REPS$(RESET)=$(DIM)3$(RESET)                                 Repetitions per model/scenario"
	@echo -e "    $(YELLOW)JUDGE$(RESET)=$(DIM)\"google/gemini-3-flash-preview\"$(RESET) Judge model"
	@echo -e "    $(YELLOW)WORKERS$(RESET)=$(DIM)6$(RESET)                               Max parallel workers"
	@echo -e "    $(YELLOW)FILE$(RESET)=$(DIM)tests/test_runner_cli_probe.py$(RESET)     Single test file"
	@echo -e "    $(YELLOW)K$(RESET)=$(DIM)test_something$(RESET)                       pytest -k pattern"
	@echo ""
	@echo -e "  $(BOLD)$(CYAN)Examples$(RESET)"
	@echo -e "    $(DIM)make run MODELS=\"x-ai/grok-4.1-fast\" REPS=3$(RESET)"
	@echo -e "    $(DIM)make rejudge MODELS=\"openai/gpt-oss-20b\"$(RESET)"
	@echo -e "    $(DIM)make probe MODELS=\"deepseek/deepseek-v4-flash\"$(RESET)"
	@echo -e "    $(DIM)make test-file FILE=tests/test_outcome_classification.py$(RESET)"
	@echo -e "    $(DIM)make test-match K=test_rejudge$(RESET)"
	@echo ""

# ─── Setup ──────────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install project + test dependencies in editable mode
	$(PIP) install -e ".[test]"

.PHONY: venv
venv: ## Create virtualenv and install everything
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e ".[test]"
	@test -f .env || (cp .env.example .env && echo -e "$(GREEN)Created .env from .env.example — edit it with your API key.$(RESET)")
	@echo ""
	@echo -e "$(GREEN)$(BOLD)Virtualenv ready!$(RESET) Activate with:"
	@echo -e "  $(CYAN)source $(VENV_DIR)/bin/activate$(RESET)"

.PHONY: env
env: ## Copy .env.example → .env (if .env is missing)
	@test -f .env && echo -e "$(YELLOW).env already exists — skipping$(RESET)" || \
		(cp .env.example .env && echo -e "$(GREEN)Created .env from .env.example — edit it with your API key.$(RESET)")

# ─── Benchmark ──────────────────────────────────────────────────────────────

.PHONY: run
run: ## Run benchmark (skips cached runs)
	$(CLI) run $(_models) $(_scenarios) $(_reps) $(_judge) $(_workers)

.PHONY: rerun
rerun: ## Re-run benchmark ignoring cache (costly!)
	@echo -e "$(YELLOW)$(BOLD)WARNING:$(RESET)$(YELLOW) This ignores all cache and re-runs from scratch.$(RESET)"
	$(CLI) rerun $(_models) $(_scenarios) $(_reps) $(_judge) $(_workers)

.PHONY: run-sequential
run-sequential: ## Run benchmark without parallelism
	$(CLI) run --no-parallel $(_models) $(_scenarios) $(_reps) $(_judge)

.PHONY: report
report: ## Regenerate reports from cached results
	$(CLI) report

.PHONY: rejudge
rejudge: ## Re-evaluate cached runs with the judge (cheap)
	$(CLI) rejudge $(_models) $(_scenarios) $(_reps) $(_judge)

.PHONY: probe
probe: ## Probe models for reasoning format
	$(CLI) probe $(_models)

.PHONY: probe-force
probe-force: ## Probe models, ignoring cached probes
	$(CLI) probe --force $(_models)

# ─── Testing ────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run full test suite with coverage
	$(PYTEST)

.PHONY: test-quick
test-quick: ## Run tests without coverage (faster)
	$(PYTEST) --no-cov -q

.PHONY: test-file
test-file: ## Run a single test file (FILE=tests/test_foo.py)
	@test -n "$(FILE)" || (echo -e "$(RED)ERROR: specify FILE=tests/test_foo.py$(RESET)" && exit 1)
	$(PYTEST) --no-cov "$(FILE)"

K ?=
.PHONY: test-match
test-match: ## Run tests matching a pattern (K=test_something)
	@test -n "$(K)" || (echo -e "$(RED)ERROR: specify K=test_something$(RESET)" && exit 1)
	$(PYTEST) --no-cov -k "$(K)"

.PHONY: coverage
coverage: ## Generate HTML coverage report and open it
	$(PYTEST) --cov-report=html
	@echo -e "$(GREEN)HTML report at htmlcov/index.html$(RESET)"

# ─── Quality ────────────────────────────────────────────────────────────────

.PHONY: check
check: test lint ## Run all checks (tests + lint)
	@echo -e "$(GREEN)$(BOLD)All checks passed.$(RESET)"

.PHONY: lint
lint: ## Run ruff linter
	@command -v ruff >/dev/null 2>&1 && ruff check src/ tests/ || \
		echo -e "$(YELLOW)ruff not installed — skipping. Install with: pip install ruff$(RESET)"

.PHONY: format
format: ## Auto-format code with ruff
	@command -v ruff >/dev/null 2>&1 && ruff format src/ tests/ || \
		echo -e "$(YELLOW)ruff not installed — skipping. Install with: pip install ruff$(RESET)"

.PHONY: typecheck
typecheck: ## Run mypy type checking
	@command -v mypy >/dev/null 2>&1 && mypy src/ || \
		echo -e "$(YELLOW)mypy not installed — skipping. Install with: pip install mypy$(RESET)"

# ─── Logs & Info ────────────────────────────────────────────────────────────

.PHONY: logs
logs: ## List recent log files
	$(CLI) logs

.PHONY: logs-tail
logs-tail: ## Show last 50 lines of latest log
	$(CLI) logs --tail

.PHONY: models
models: ## List all active models from the registry
	@$(PYTHON) -c "$$MODELS_SCRIPT"

.PHONY: leaderboard
leaderboard: ## Display the current leaderboard file
	@test -f results/LEADERBOARD.md && cat results/LEADERBOARD.md || \
		echo -e "$(YELLOW)No leaderboard yet. Run 'make run' or 'make report' first.$(RESET)"

.PHONY: cli-help
cli-help: ## Show the full CLI help tree
	@$(CLI) --help
	@echo ""
	@echo -e "$(DIM)──── run ────$(RESET)"
	@$(CLI) run --help 2>/dev/null | tail -n +5
	@echo -e "$(DIM)──── rerun ────$(RESET)"
	@$(CLI) rerun --help 2>/dev/null | tail -n +5
	@echo -e "$(DIM)──── report ────$(RESET)"
	@$(CLI) report --help 2>/dev/null | tail -n +5
	@echo -e "$(DIM)──── rejudge ────$(RESET)"
	@$(CLI) rejudge --help 2>/dev/null | tail -n +5
	@echo -e "$(DIM)──── probe ────$(RESET)"
	@$(CLI) probe --help 2>/dev/null | tail -n +5
	@echo -e "$(DIM)──── logs ────$(RESET)"
	@$(CLI) logs --help 2>/dev/null | tail -n +5

# ─── Maintenance ────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove build artifacts and Python caches
	rm -rf build/ dist/ *.egg-info .pytest_cache htmlcov .coverage coverage.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-logs
clean-logs: ## Remove all log files
	rm -rf logs/

.PHONY: clean-probes
clean-probes: ## Remove cached probe results
	rm -rf probes/

.PHONY: clean-all
clean-all: clean clean-logs clean-probes ## Remove all generated artifacts
	@echo -e "$(GREEN)Cleaned build artifacts, logs, and probes.$(RESET)"
	@echo -e "$(DIM)Note: cache/ and results/ are preserved. Delete manually if needed.$(RESET)"
