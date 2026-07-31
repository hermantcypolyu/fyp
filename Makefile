# rl-coding-agents - developer entry points.
#
# Everything here is CPU-only and safe to run on any machine.
# There is deliberately NO training target: see AGENT_BUILD_SPEC.md 0.2.
#
# Windows users without GNU make: run `./make.ps1 <target>` instead.

UV ?= uv
PY := $(UV) run python

.DEFAULT_GOAL := help
.PHONY: help install new-task gold-patches validate verify-gold catalogue smoke smoke-gym test lint fmt check clean freshclone

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install everything (core + dev)
	$(UV) sync --extra dev

new-task: ## Scaffold a blank task: make new-task ID=my_task CATEGORY=bug_fix
	$(PY) benchmark/tools/new_task.py $(ID) --category $(CATEGORY) $(if $(SPLIT),--split $(SPLIT),) $(if $(DIFFICULTY),--difficulty $(DIFFICULTY),)

gold-patches: ## Regenerate solution/gold.patch for every task from solution/fixed/
	$(PY) benchmark/tools/make_gold_patch.py --all

validate: ## Schema-validate every task in benchmark/tasks/**
	$(PY) benchmark/schema/validate_tasks.py --all

verify-gold: ## Prove every gold patch fixes its task and the untouched repo fails
	$(PY) scripts/verify_gold_patches.py --split all

catalogue: ## Regenerate benchmark/catalogue.md from task.yaml files
	$(PY) benchmark/tools/build_catalogue.py

smoke: ## Run ONE task end-to-end through the environment and print the reward
	$(PY) scripts/smoke_one_task.py --limit 1

test: ## Unit tests (rewards, metrics, sandbox isolation, schema)
	$(UV) run pytest

lint: ## Ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: ## Auto-fix lint and format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

check: validate verify-gold test lint ## Everything CI runs

smoke-gym: ## Same smoke test, but over real NeMo Gym HTTP servers (needs `uv sync --extra gym`, Linux/macOS)
	$(PY) scripts/smoke_one_task.py --limit 1 --transport gym

freshclone: ## The five-step acceptance check from AGENT_BUILD_SPEC 0.3
	$(UV) sync --extra dev
	$(MAKE) validate
	$(MAKE) verify-gold
	$(MAKE) test
	$(MAKE) smoke

clean: ## Remove caches and sandbox scratch
	$(PY) -c "import shutil,pathlib;[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PY) -c "import shutil;[shutil.rmtree(p,ignore_errors=True) for p in ['.pytest_cache','.ruff_cache','.sandbox']]"
