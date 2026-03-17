# Multi-Agent Report System Design Notes

## 1. Module Positioning

This module sits at the final layer of the multi-agent data science workflow.

It does not train models, and it does not read teammates' notebooks.  
Its only responsibility is to read structured JSON delivered by upstream teammates and generate two final deliverables:

- a technical report
- a business action recommendation report

Overall workflow:

```text
Upstream agents / teammates complete data processing and modeling
                ↓
          Output a standard JSON file
                ↓
        Report generation module (this module)
         ├─ Technical report
         └─ Business report
```

## 2. Why This Module Is Separated

This separation has three advantages:

- upstream modeling workflows are decoupled from downstream reporting
- the same reporting logic can be reused across different projects
- results no longer stop at “model scores” and can be turned into formal deliverables and action plans

## 3. Input and Output

Input:

- structured JSON delivered by upstream teammates

The current module supports two schemas:

- legacy: `meta / data_understanding / data_cleaning / feature_engineering / modeling / evaluation / business_context`
- new: `project_info / dataset_summary / pipeline_trace / model_results / interpretability / risk_scoring / business_constraints`

Recommended team standard:

- `team_json_template.json`

Output:

- `technical_report_*.md`
- `business_report_*.md`

## 4. Module Capabilities

The current version already supports:

- reading and validating input JSON
- supporting both legacy and new JSON schemas
- adapting the report topic automatically from project metadata
- generating technical reports
- generating business reports
- saving Markdown output

## 5. Value Within the Overall Project

Without this module, the system usually only outputs:

- which model performs best
- what the metrics are
- which features matter most

With this module, the system can continue all the way to:

```text
Data analysis -> Model evaluation -> Result interpretation -> Action recommendation -> Formal report
```

This is also the module’s core contribution to the overall project:  
it turns technical results into conclusions that teachers, teammates, or business stakeholders can directly use.
