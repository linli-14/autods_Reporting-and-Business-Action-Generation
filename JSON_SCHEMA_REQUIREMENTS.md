# JSON Schema Requirements

This document defines the recommended JSON input format for the report-generation module.

The goal is simple:

- upstream teammates produce one structured JSON file
- this module reads that JSON
- the system generates a technical report and a business report

The repository currently supports:

- old schema: `meta / data_understanding / data_cleaning / feature_engineering / modeling / evaluation / business_context`
- new schema: `project_info / dataset_summary / pipeline_trace / model_results / interpretability / risk_scoring / business_constraints`

For all new teamwork, use the new schema in [team_json_template.json](/Users/lin/UTS/36103 statistical/assignment/assignment2/OneDrive_1_16-05-2025/project_folder/team_json_template.json).

## Quick Handoff Checklist

Before handing JSON to the report module, confirm:

- the file is valid JSON and can be opened normally
- all required top-level sections are present
- numeric metrics are numbers, not percentage strings
- `selected_model.metrics` is filled
- `feature_importance` is present
- `risk_summary` is present
- if available, `high_risk_instances` is included
- if available, budget or intervention fields are included for better business recommendations

If the team does not know where to start, copy [team_json_template.json](/Users/lin/UTS/36103 statistical/assignment/assignment2/OneDrive_1_16-05-2025/project_folder/team_json_template.json) and only replace values.

## Required Sections

These top-level sections are required in the new schema:

- `project_info`
- `dataset_summary`
- `pipeline_trace`
- `model_results`
- `interpretability`
- `risk_scoring`
- `business_constraints`

Optional top-level section:

- `reporting_preferences`

Recommended metadata:

- `schema_version`
- `generated_at`

## Section Requirements

### `project_info`

Purpose:
Defines what the project is about, what is being predicted, and why it matters.

Required fields:

- `project_name`
- `problem_type`
- `business_objective`
- `target_variable`
- `target_definition`

Recommended fields:

- `project_id`
- `industry`
- `stakeholders`
- `target_audience`
- `success_criteria`

Example:

```json
{
  "project_name": "Customer Churn Risk Prediction",
  "problem_type": "binary_classification",
  "business_objective": "Identify customers at risk of churn and prioritize retention actions",
  "target_variable": "churn_risk",
  "target_definition": "1 means the customer is likely to churn within the next 30 days"
}
```

Notes:

- `problem_type` should be explicit, such as `binary_classification`, `multiclass_classification`, or `regression`.
- `target_definition` should explain what the target label actually means in business terms.

### `dataset_summary`

Purpose:
Provides the data overview needed for the technical report.

Required fields:

- `dataset_name`
- `num_rows`
- `num_features`

Recommended fields:

- `data_source`
- `num_rows_after_cleaning`
- `feature_types`
- `missing_value_summary`
- `class_distribution`
- `class_imbalance_ratio`
- `data_quality_score`
- `data_quality_notes`
- `key_insights`

Notes:

- For classification tasks, include `class_distribution` whenever possible.
- `feature_types` can be counts or a more detailed object, but keep it consistent within the team.

### `pipeline_trace`

Purpose:
Explains what happened upstream before report generation.

Required fields:

- `data_cleaning`
- `feature_engineering`
- `evaluation_setup`

Recommended fields:

- `model_selection`

Suggested internal fields:

- `data_cleaning.performed`
- `data_cleaning.actions`
- `feature_engineering.performed`
- `feature_engineering.actions`
- `evaluation_setup.primary_metric`
- `evaluation_setup.cross_validation`
- `evaluation_setup.train_test_split`

Notes:

- This section does not need every notebook detail.
- Keep it concise and structured.

### `model_results`

Purpose:
Stores model comparison results and the final selected model.

Required fields:

- `candidate_models`
- `selected_model`

Required inside `selected_model`:

- `model_name`
- `selection_reason`
- `metrics`

Recommended metric fields for classification:

- `accuracy`
- `precision`
- `recall`
- `f1_score`
- `roc_auc`

Recommended metric fields for regression:

- `mae`
- `rmse`
- `r2`

Recommended extra fields:

- `selected_model.confusion_matrix`
- `selected_model.cv_scores`
- `selected_model.cv_mean`
- `selected_model.cv_std`
- `baseline_score`
- `improvement_over_baseline`
- `key_insights`

Notes:

- Do not provide only the best model. Include at least 2 candidate models if comparison exists.
- `selection_reason` should be written in plain language, not just “best performance”.

### `interpretability`

Purpose:
Explains why the model makes its predictions and what variables matter most.

Required fields:

- `feature_importance`

Recommended fields:

- `local_explanations_available`
- `explanation_method`
- `key_insights`

Required inside each feature item:

- `feature`
- `importance`

Recommended inside each feature item:

- `direction`

Notes:

- `direction` is very helpful for the business report, for example:
  `lower activity increases churn risk`
- Try to provide the top 3 to top 10 most important features.

### `risk_scoring`

Purpose:
Provides individual-level or case-level risk information for actionable recommendations.

Required fields:

- `risk_threshold`
- `risk_summary`

Recommended fields:

- `risk_bands`
- `high_risk_instances`

Recommended inside each high-risk instance:

- `case_id`
- `predicted_probability`
- `predicted_label`
- `top_risk_factors`
- `suggested_action_tags`

Notes:

- This is one of the most valuable sections for the business report.
- Use a generic identifier like `case_id` unless the project requires a domain-specific ID.
- If privacy matters, do not put names or sensitive personal identifiers here.

### `business_constraints`

Purpose:
Defines real-world constraints so the system can translate analysis into prioritized action.

Required fields:

- `preferred_strategy`

Recommended fields:

- `intervention_budget`
- `max_cases_for_immediate_intervention`
- `intervention_cost_per_case`
- `estimated_cost_of_missing_case`
- `available_interventions`

Recommended inside `available_interventions`:

- `action`
- `cost`
- `expected_impact`

Notes:

- This section helps the business report explain trade-offs and priorities.
- If no budget information is available, the system can still work, but ROI discussion will be weaker.

### `reporting_preferences`

Purpose:
Optional controls for formatting and output style.

Optional fields:

- `technical_report_format`
- `business_report_format`
- `language`
- `include_visual_placeholders`
- `max_business_report_length`
- `tone`

Notes:

- This section is optional.
- If omitted, the generator should still run.

## Minimal Viable JSON

If the team wants the smallest workable input for the new schema, keep at least:

- `project_info`
- `dataset_summary`
- `pipeline_trace`
- `model_results`
- `interpretability`
- `risk_scoring`
- `business_constraints`

But for higher-quality reports, include the recommended fields too.

## Type Expectations

These are practical expectations rather than strict JSON Schema rules:

- counts like `num_rows`, `num_features` should be numbers
- metric values like `accuracy`, `f1_score`, `roc_auc`, `mae`, `rmse`, `r2` should be numbers
- list fields like `stakeholders`, `actions`, `key_insights`, `top_risk_factors` should be arrays
- objects like `metrics`, `confusion_matrix`, `risk_summary` should remain objects, not strings

Good:

```json
"f1_score": 0.879
```

Bad:

```json
"f1_score": "87.9%"
```

The system works better when metrics are numeric.

## Common Errors

### 1. Missing top-level sections

Problem:
The file omits one of the required sections such as `model_results` or `risk_scoring`.

Result:
The report generator may fail validation or produce weak output.

Fix:
Always start from [team_json_template.json](/Users/lin/UTS/36103 statistical/assignment/assignment2/OneDrive_1_16-05-2025/project_folder/team_json_template.json).

### 2. Only overall scores, no selected model details

Problem:
The JSON only includes one summary metric like F1-score.

Result:
The technical report lacks model comparison and selection rationale.

Fix:
Include `candidate_models`, `selected_model`, and `selection_reason`.

### 3. No interpretability section

Problem:
The JSON says which cases are high risk, but not why.

Result:
The business report can rank cases but cannot explain the action logic well.

Fix:
Include at least top feature importances and simple direction text.

### 4. No risk-level instances

Problem:
The JSON only includes aggregate metrics.

Result:
The business report can discuss performance, but not who should be prioritized.

Fix:
Include `high_risk_instances` when available.

### 5. Numbers stored as strings

Problem:
Metrics or counts are stored like `"0.88"` or `"88%"`.

Result:
This can break calculations or weaken interpretation.

Fix:
Use numeric JSON values whenever possible.

### 6. Domain-specific IDs hardcoded everywhere

Problem:
The schema uses fields like `student_id` even when the project is not about students.

Result:
The template becomes hard to reuse across datasets.

Fix:
Prefer generic names like `case_id`, unless the project truly requires a domain-specific field.

### 7. Too much notebook detail

Problem:
The upstream team dumps every preprocessing step and every intermediate artifact into JSON.

Result:
The file becomes noisy and harder to maintain.

Fix:
Only include information needed for report generation and downstream explanation.

## Team Workflow Recommendation

Recommended team process:

1. upstream teammates export results using [team_json_template.json](/Users/lin/UTS/36103 statistical/assignment/assignment2/OneDrive_1_16-05-2025/project_folder/team_json_template.json)
2. they validate required sections before handoff
3. this module reads the JSON and generates the reports

Suggested handoff rule:

- if a field is unknown, leave it empty only if it is optional
- do not rename required fields casually
- do not replace arrays or objects with free-text paragraphs

## Final Recommendation

For all future projects:

- treat `team_json_template.json` as the official team interface
- keep field names stable across datasets
- only change values and project-specific content

That way, the same report generator can keep working even when the dataset and business domain change.
