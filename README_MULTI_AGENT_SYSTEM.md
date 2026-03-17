# Multi-Agent Report Generation System

This document focuses only on usage.  
For project background and repository positioning, see [README.md](/Users/lin/UTS/36103 statistical/assignment/assignment2/OneDrive_1_16-05-2025/project_folder/README.md). For JSON field requirements, see [JSON_SCHEMA_REQUIREMENTS.md](/Users/lin/UTS/36103 statistical/assignment/assignment2/OneDrive_1_16-05-2025/project_folder/JSON_SCHEMA_REQUIREMENTS.md).

## 1. Input and Output

Input:

- Structured JSON delivered by upstream teammates

Output:

- `reports/technical_report_YYYYMMDD_HHMMSS.md`
- `reports/business_report_YYYYMMDD_HHMMSS.md`

Recommended template for teammates:

- [team_json_template.json](/Users/lin/UTS/36103 statistical/assignment/assignment2/OneDrive_1_16-05-2025/project_folder/team_json_template.json)

## 2. Supported JSON Schemas

The current program supports two input schemas:

- Legacy schema: `meta / data_understanding / data_cleaning / feature_engineering / modeling / evaluation / business_context`
- New schema: `project_info / dataset_summary / pipeline_trace / model_results / interpretability / risk_scoring / business_constraints`

For future work, the team should standardize on the new schema. The legacy schema is kept only for backward compatibility.

## 3. Install Dependencies

```bash
python3.11 -m venv .venv311
. .venv311/bin/activate
pip install -r requirements_langchain.txt
```

## 4. Configure the API Key

```bash
cp .env.example .env
```

In `.env`, set:

```env
OPENAI_API_KEY=sk-your-openai-key
```

If you are using an OpenAI-compatible provider such as NVIDIA NIM, you must also set:

```env
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1
OPENAI_MODEL=minimaxai/minimax-m2.5
```

The safest approach is to copy from [`.env.example`](/Users/lin/UTS/36103 statistical/assignment/assignment2/OneDrive_1_16-05-2025/project_folder/.env.example).

## 5. Run Commands

Generate both reports:

```bash
.venv311/bin/python multi_agent_report_generator.py --json your_team_output.json
```

Generate only the technical report:

```bash
.venv311/bin/python multi_agent_report_generator.py --json your_team_output.json --mode technical
```

Generate only the business report:

```bash
.venv311/bin/python multi_agent_report_generator.py --json your_team_output.json --mode business
```

Specify a custom output directory:

```bash
.venv311/bin/python multi_agent_report_generator.py --json your_team_output.json --output-dir demo_reports
```

## 6. Python Usage

```python
from multi_agent_report_generator import MultiAgentReportGenerator

generator = MultiAgentReportGenerator()
reports = generator.generate_reports("your_team_output.json")
```

## 7. Troubleshooting

### 1. JSON fields are incomplete

Check first:

- whether the file follows [team_json_template.json](/Users/lin/UTS/36103 statistical/assignment/assignment2/OneDrive_1_16-05-2025/project_folder/team_json_template.json)
- whether it matches [JSON_SCHEMA_REQUIREMENTS.md](/Users/lin/UTS/36103 statistical/assignment/assignment2/OneDrive_1_16-05-2025/project_folder/JSON_SCHEMA_REQUIREMENTS.md)

### 2. OpenAI-related errors

Check:

- whether `OPENAI_API_KEY` in `.env` is valid
- whether the account still has available quota
- whether the current network connection is working

### 3. Want to demo the flow first

Run:

```bash
.venv311/bin/python multi_agent_report_generator.py --json example_pipeline_output.json
```
