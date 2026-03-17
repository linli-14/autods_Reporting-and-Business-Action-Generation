# Multi-Agent Report Generator

This repository contains the report-generation layer of a multi-agent data science workflow.

Its job is simple:

1. upstream teammates or agents produce a structured JSON output
2. this module reads that JSON
3. it generates a technical report and a business-facing report in Markdown

The current version is designed for reusable, cross-project use. It is no longer tied to a specific notebook pipeline or a single dataset theme such as student dropout prediction.

## What This Repo Focuses On

- JSON-in, report-out workflow
- OpenAI-powered technical report generation
- OpenAI-powered business action translation
- JSON structure validation before report generation
- Project-theme adaptation based on metadata in the input JSON

## Role In The Full System

This module is the final reporting layer in the larger multi-agent workflow.

Its responsibility is not model training or notebook parsing. Instead, it:

- receives a standardized JSON output from upstream teammates or agents
- generates a technical report for technical readers
- generates a business report for non-technical stakeholders

Recommended team handoff:

```text
upstream modeling/evaluation -> structured JSON -> this repo -> final reports
```

## What Has Already Been Built

- direct report generation from structured JSON
- two report modes: technical and business
- OpenAI integration through `OPENAI_API_KEY`
- input validation for required JSON sections
- topic adaptation using project metadata such as dataset, target, use case, and audience

## Quick Start

```bash
python3.11 -m venv .venv311
. .venv311/bin/activate
pip install -r requirements_langchain.txt
cp .env.example .env
```

Set your OpenAI key in `.env`:

```env
OPENAI_API_KEY=sk-your-openai-key
```

Run the example:

```bash
.venv311/bin/python multi_agent_report_generator.py --json example_pipeline_output.json
```

Outputs will be saved to `reports/`.

## Expected Input

The generator expects a structured JSON file from upstream teammates or agents.

Required top-level sections:

- `meta`
- `data_understanding`
- `data_cleaning`
- `feature_engineering`
- `modeling`
- `evaluation`
- `business_context`

Recommended extra fields for better topic adaptation:

- `meta.project_theme`
- `meta.project_description`
- `meta.target_variable`
- `business_context.use_case`
- `business_context.industry`
- `business_context.target_audience`
- `business_context.stakeholders`
- `business_context.business_goal`
- `business_context.project_objective`

## Repository Structure

Core files:

- `README.md`: GitHub homepage and project overview
- `README_MULTI_AGENT_SYSTEM.md`: detailed usage guide
- `JSON_SCHEMA_REQUIREMENTS.md`: field-level requirements for teammate JSON handoff
- `multi_agent_report_generator.py`: main report generator
- `example_pipeline_output.json`: example upstream JSON
- `team_json_template.json`: recommended JSON interface for teammates
- `requirements_langchain.txt`: dependencies
- `quick_start.sh`: quick setup and run script
- `test_system.py`: local structure validation
- `reports/`: sample generated outputs

Supporting project context:

- `plans/multi_agent_report_system_design.md`: compact design note for group integration

Research assets kept in the repo:

- `*.ipynb`
- `train_set.csv`
- `test_set.csv`
- `data.csv`

These research assets are preserved as project context, but the report generator does not depend on them anymore.

## Suggested GitHub Presentation

If you are sharing this repo with teammates or markers, the best reading order is:

1. `README.md`
2. `README_MULTI_AGENT_SYSTEM.md`
3. `multi_agent_report_generator.py`
4. `example_pipeline_output.json`
5. files under `reports/`

## Current Scope

- Provider: OpenAI only
- Input source: structured JSON only
- Output format: Markdown reports
- Primary use case: final reporting layer in a multi-agent workflow

## Team Contract

Upstream teammates only need to do three things:

1. output a standardized JSON file
2. include the required top-level sections
3. pass that JSON file into this report module

This keeps the collaboration boundary clear and avoids depending on teammates' notebooks.

## Notes

- This repo intentionally avoids depending on teammates' notebooks.
- If upstream JSON is missing required fields, the program will fail early with a clear validation message.
- The prompt is now designed to adapt to different project themes based on JSON metadata rather than assuming a fixed domain.
