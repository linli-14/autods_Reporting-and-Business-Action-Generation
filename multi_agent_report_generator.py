#!/usr/bin/env python3
"""
多智能体报告生成系统
基于 LangChain 构建，用于接收上游智能体输出的结构化 JSON，
并自动生成技术报告和业务建议。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()


class MultiAgentReportGenerator:
    """多智能体报告生成器（仅支持 OpenAI）"""

    REQUIRED_SECTIONS = [
        "meta",
        "data_understanding",
        "data_cleaning",
        "feature_engineering",
        "modeling",
        "evaluation",
        "business_context",
    ]
    NEW_SCHEMA_SECTIONS = [
        "project_info",
        "dataset_summary",
        "pipeline_trace",
        "model_results",
        "interpretability",
        "risk_scoring",
        "business_constraints",
    ]
    LANGUAGE_ALIASES = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese",
        "cn": "Chinese",
        "en": "English",
        "en-us": "English",
        "en-gb": "English",
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        output_dir: str = "reports",
    ):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if self._is_placeholder_key(self.api_key):
            raise ValueError("请在.env中配置真实的 OPENAI_API_KEY")

        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.env_report_language = os.getenv("REPORT_LANGUAGE")
        self.default_report_language = self.env_report_language or "en"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.llm_technical = ChatOpenAI(
            model=self.model_name,
            temperature=0.3,
            openai_api_key=self.api_key,
        )
        self.llm_business = ChatOpenAI(
            model=self.model_name,
            temperature=0.5,
            openai_api_key=self.api_key,
        )

        self._setup_chains()

    @staticmethod
    def _is_placeholder_key(key: str) -> bool:
        if not key:
            return True
        normalized = key.strip().lower()
        return (
            normalized.endswith("_here")
            or "your_" in normalized
            or "api_key_here" in normalized
        )

    def _setup_chains(self) -> None:
        technical_prompt = PromptTemplate(
            input_variables=["json_data", "project_context", "report_language_name"],
            template=self._get_technical_report_prompt(),
        )
        self.technical_chain = LLMChain(
            llm=self.llm_technical,
            prompt=technical_prompt,
            output_key="technical_report",
            verbose=True,
        )

        business_prompt = PromptTemplate(
            input_variables=[
                "json_data",
                "technical_report",
                "project_context",
                "report_language_name",
            ],
            template=self._get_business_translation_prompt(),
        )
        self.business_chain = LLMChain(
            llm=self.llm_business,
            prompt=business_prompt,
            output_key="business_report",
            verbose=True,
        )

        self.full_pipeline = SequentialChain(
            chains=[self.technical_chain, self.business_chain],
            input_variables=["json_data", "project_context", "report_language_name"],
            output_variables=["technical_report", "business_report"],
            verbose=True,
        )

    def validate_input_json(self, json_data: Dict) -> None:
        """校验上游智能体交付的 JSON 是否满足最小输入要求。"""
        missing_sections = [
            section for section in self.REQUIRED_SECTIONS if section not in json_data
        ]
        if missing_sections:
            raise ValueError(
                "输入JSON缺少必需字段: "
                + ", ".join(missing_sections)
                + "。请让上游智能体按约定结构输出。"
            )

    @classmethod
    def _is_current_schema(cls, json_data: Dict[str, Any]) -> bool:
        return all(section in json_data for section in cls.REQUIRED_SECTIONS)

    @classmethod
    def _is_new_schema(cls, json_data: Dict[str, Any]) -> bool:
        return all(section in json_data for section in cls.NEW_SCHEMA_SECTIONS)

    @staticmethod
    def _coerce_list(value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _extract_metric_value(metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        value = metrics.get(metric_name)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _compute_class_imbalance_ratio(class_distribution: Dict[str, Any]) -> Optional[float]:
        values = [
            float(value)
            for value in class_distribution.values()
            if isinstance(value, (int, float)) and float(value) > 0
        ]
        if len(values) < 2:
            return None
        return round(max(values) / min(values), 3)

    @staticmethod
    def _normalize_confusion_matrix(confusion_matrix: Dict[str, Any]) -> Dict[str, Any]:
        if not confusion_matrix:
            return {}
        if {"tn", "fp", "fn", "tp"}.issubset(confusion_matrix.keys()):
            return {
                "true_negative": confusion_matrix.get("tn"),
                "false_positive": confusion_matrix.get("fp"),
                "false_negative": confusion_matrix.get("fn"),
                "true_positive": confusion_matrix.get("tp"),
            }
        return confusion_matrix

    @staticmethod
    def _normalize_operations(actions: Any) -> list:
        normalized = []
        for index, item in enumerate(MultiAgentReportGenerator._coerce_list(actions), start=1):
            if isinstance(item, dict):
                normalized.append(
                    {
                        "operation": item.get("operation", f"step_{index}"),
                        "detail": item.get("detail", item.get("action", "")),
                        "affected_columns": item.get("affected_columns", ""),
                    }
                )
            else:
                normalized.append(
                    {
                        "operation": f"step_{index}",
                        "detail": str(item),
                        "affected_columns": "",
                    }
                )
        return normalized

    def _normalize_new_schema(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        project_info = json_data.get("project_info", {})
        dataset_summary = json_data.get("dataset_summary", {})
        pipeline_trace = json_data.get("pipeline_trace", {})
        model_results = json_data.get("model_results", {})
        interpretability = json_data.get("interpretability", {})
        risk_scoring = json_data.get("risk_scoring", {})
        business_constraints = json_data.get("business_constraints", {})
        reporting_preferences = json_data.get("reporting_preferences", {})

        candidate_models = self._coerce_list(model_results.get("candidate_models"))
        selected_model = model_results.get("selected_model", {})
        selected_metrics = selected_model.get("metrics", {})
        primary_metric = (
            pipeline_trace.get("evaluation_setup", {}).get("primary_metric")
            or model_results.get("primary_metric")
            or "f1_score"
        )
        primary_score = self._extract_metric_value(selected_metrics, primary_metric)

        class_distribution = dataset_summary.get("class_distribution", {})
        feature_importance = self._coerce_list(
            interpretability.get("feature_importance")
            or interpretability.get("feature_importances")
        )

        normalized_models = []
        for rank, model in enumerate(candidate_models, start=1):
            metrics = model.get("metrics", {})
            normalized_models.append(
                {
                    "name": model.get("model_name", model.get("name", f"Model {rank}")),
                    "rank": model.get("rank", rank),
                    **metrics,
                }
            )

        business_goal = project_info.get("business_objective")
        if isinstance(project_info.get("success_criteria"), list):
            success_summary = "; ".join(str(item) for item in project_info["success_criteria"])
        else:
            success_summary = None

        normalized = {
            "meta": {
                "schema_version": json_data.get("schema_version", "2.0"),
                "pipeline_id": project_info.get("project_id", project_info.get("project_name", "team_project")),
                "dataset_name": dataset_summary.get("dataset_name", project_info.get("project_name", "Unknown Project")),
                "dataset_source": dataset_summary.get("data_source"),
                "target_variable": project_info.get("target_variable"),
                "task_type": project_info.get("problem_type"),
                "project_theme": project_info.get("project_name"),
                "project_description": project_info.get("target_definition"),
                "timestamp": json_data.get("generated_at"),
                "models_evaluated": len(candidate_models),
            },
            "data_understanding": {
                "n_rows": dataset_summary.get("num_rows"),
                "n_cols": dataset_summary.get("num_features"),
                "n_rows_after_cleaning": dataset_summary.get(
                    "num_rows_after_cleaning",
                    dataset_summary.get("num_rows"),
                ),
                "feature_types": dataset_summary.get("feature_types", {}),
                "class_distribution": class_distribution,
                "class_imbalance_ratio": dataset_summary.get("class_imbalance_ratio")
                or self._compute_class_imbalance_ratio(class_distribution),
                "missing_values_summary": dataset_summary.get(
                    "missing_value_summary",
                    {},
                ),
                "key_insights": self._coerce_list(dataset_summary.get("key_insights")),
            },
            "data_cleaning": {
                "operations_performed": self._normalize_operations(
                    pipeline_trace.get("data_cleaning", {}).get("actions")
                ),
                "outliers_detected": self._coerce_list(
                    pipeline_trace.get("data_cleaning", {}).get("outliers_detected")
                ),
                "data_quality_score": dataset_summary.get("data_quality_score"),
                "quality_notes": dataset_summary.get("data_quality_notes")
                or pipeline_trace.get("data_cleaning", {}).get("notes"),
            },
            "feature_engineering": {
                "features_created": self._coerce_list(
                    pipeline_trace.get("feature_engineering", {}).get("features_created")
                ),
                "features_dropped": self._coerce_list(
                    pipeline_trace.get("feature_engineering", {}).get("features_dropped")
                ),
                "encoding_applied": pipeline_trace.get("feature_engineering", {}).get(
                    "encoding_applied",
                    {},
                ),
                "feature_importances": feature_importance,
                "final_feature_count": dataset_summary.get(
                    "num_features_after_engineering",
                    dataset_summary.get("num_features"),
                ),
                "key_insights": self._coerce_list(
                    interpretability.get("key_insights")
                ),
            },
            "modeling": {
                "best_model": {
                    "name": selected_model.get("model_name", selected_model.get("name")),
                    "params": selected_model.get("params", {}),
                    "training_time_seconds": selected_model.get("training_time_seconds"),
                    "optimization_method": pipeline_trace.get("model_selection", {}).get(
                        "selection_strategy"
                    ),
                },
                "models_compared": normalized_models,
                "selection_reason": selected_model.get("selection_reason")
                or model_results.get("selection_reason"),
            },
            "evaluation": {
                "primary_metric": primary_metric,
                "primary_score": primary_score,
                "metrics": selected_metrics,
                "confusion_matrix": self._normalize_confusion_matrix(
                    selected_model.get("confusion_matrix", {})
                ),
                "cv_scores": self._coerce_list(
                    selected_model.get("cv_scores")
                    or model_results.get("cv_scores")
                ),
                "cv_mean": selected_model.get("cv_mean") or model_results.get("cv_mean"),
                "cv_std": selected_model.get("cv_std") or model_results.get("cv_std"),
                "baseline_score": model_results.get("baseline_score"),
                "improvement_over_baseline": model_results.get(
                    "improvement_over_baseline"
                ),
                "key_insights": self._coerce_list(model_results.get("key_insights")),
            },
            "business_context": {
                "industry": project_info.get("industry"),
                "use_case": business_goal,
                "target_audience": project_info.get("target_audience")
                or ", ".join(str(item) for item in self._coerce_list(project_info.get("stakeholders"))),
                "stakeholders": self._coerce_list(project_info.get("stakeholders")),
                "business_goal": business_goal,
                "project_objective": project_info.get("target_definition"),
                "report_language": reporting_preferences.get("language"),
                "report_detail_level": reporting_preferences.get(
                    "business_report_format"
                ),
                "decision_threshold": risk_scoring.get("risk_threshold"),
                "intervention_budget": business_constraints.get("intervention_budget"),
                "intervention_cost_per_student": business_constraints.get(
                    "intervention_cost_per_student",
                    business_constraints.get("intervention_cost_per_case"),
                ),
                "estimated_cost_of_missing_case": business_constraints.get(
                    "estimated_cost_of_missing_at_risk_student",
                    business_constraints.get("estimated_cost_of_missing_case"),
                ),
                "preferred_strategy": business_constraints.get("preferred_strategy"),
                "success_criteria": success_summary,
            },
            "project_info": project_info,
            "dataset_summary": dataset_summary,
            "pipeline_trace": pipeline_trace,
            "model_results": model_results,
            "interpretability": interpretability,
            "risk_scoring": risk_scoring,
            "business_constraints": business_constraints,
            "reporting_preferences": reporting_preferences,
        }
        return normalized

    def _normalize_input_json(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        if self._is_current_schema(json_data):
            normalized = json_data
        elif self._is_new_schema(json_data):
            normalized = self._normalize_new_schema(json_data)
        else:
            raise ValueError(
                "输入JSON不是受支持的结构。当前支持两种格式："
                "旧版(meta/data_understanding/...) 或新版(project_info/dataset_summary/...)。"
            )

        self.validate_input_json(normalized)
        return normalized

    def _load_json_file(self, json_file_path: str) -> Dict:
        json_path = Path(json_file_path)
        if not json_path.exists():
            raise FileNotFoundError(f"未找到JSON文件: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        return self._normalize_input_json(json_data)

    @classmethod
    def _describe_report_language(cls, report_language: Optional[str]) -> str:
        if not report_language:
            return "English"
        normalized = str(report_language).strip().lower()
        return cls.LANGUAGE_ALIASES.get(normalized, str(report_language).strip())

    def _resolve_report_language(self, json_data: Dict[str, Any]) -> str:
        business_context = json_data.get("business_context", {})
        report_language = (
            self.env_report_language
            or business_context.get("report_language")
            or self.default_report_language
        )
        business_context["report_language"] = report_language
        return report_language

    @staticmethod
    def _format_optional_context(label: str, value: Optional[object]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            if not value:
                return None
            value = ", ".join(str(item) for item in value)
        value_str = str(value).strip()
        if not value_str:
            return None
        return f"- {label}: {value_str}"

    def _build_project_context(self, json_data: Dict) -> str:
        meta = json_data.get("meta", {})
        business_context = json_data.get("business_context", {})

        context_lines = [
            self._format_optional_context("Dataset name", meta.get("dataset_name")),
            self._format_optional_context("Dataset source", meta.get("dataset_source")),
            self._format_optional_context("Task type", meta.get("task_type")),
            self._format_optional_context("Target variable", meta.get("target_variable")),
            self._format_optional_context("Project theme", meta.get("project_theme")),
            self._format_optional_context("Project description", meta.get("project_description")),
            self._format_optional_context("Use case", business_context.get("use_case")),
            self._format_optional_context("Industry", business_context.get("industry")),
            self._format_optional_context("Target audience", business_context.get("target_audience")),
            self._format_optional_context("Stakeholders", business_context.get("stakeholders")),
            self._format_optional_context(
                "Report language",
                self._describe_report_language(business_context.get("report_language")),
            ),
            self._format_optional_context(
                "Business goal", business_context.get("business_goal")
            ),
            self._format_optional_context(
                "Project objective", business_context.get("project_objective")
            ),
        ]
        usable_lines = [line for line in context_lines if line]
        if not usable_lines:
            return "- No extra project context was provided. Infer the topic strictly from the JSON content."
        return "\n".join(usable_lines)

    def _get_technical_report_prompt(self) -> str:
        return """You are a senior data scientist with more than 10 years of machine learning project experience. Your job is to write a professional technical report from structured project results.

Generate a technical report based on the project context and JSON data below.

[Project context]
{project_context}

[JSON data]
{json_data}

[Report requirements]

0. Topic adaptation:
   - Infer the project theme, industry, and task strictly from the project context and JSON content.
   - Do not assume this is an education, student, or dropout scenario unless the JSON clearly says so.
   - For classification tasks, explain metrics such as accuracy, F1, recall, precision, class imbalance, and the confusion matrix.
   - For regression tasks, focus on MAE, RMSE, R2, error range, and business impact.
   - For other task types, prioritize the core metrics that actually appear in the JSON instead of forcing a fixed template.
   - Use the entity names from the JSON for the target, risk group, and business objects.

1. Structure:
   - # Executive Summary
   - ## 1. Data Overview and Quality Assessment
   - ## 2. Feature Engineering Analysis
   - ## 3. Model Comparison and Selection
   - ## 4. Performance Evaluation
   - ## 5. Technical Recommendations and Improvement Directions

2. Writing style:
   - Professional, precise, and data-driven
   - Use concrete numbers instead of vague descriptions
   - Explain technical terms clearly while staying professional
   - State both model strengths and limitations objectively

3. Metric interpretation:
   - For classification, explain accuracy, F1 score, precision, recall, confusion matrix, and class imbalance effects
   - For regression, explain MAE, RMSE, R2, and the business meaning of prediction error
   - For all tasks, explain the practical meaning and predictive logic of the top five important features or variables
   - If cross-validation results are available, assess model stability

4. Chart recommendations:
   Describe which charts should be produced by the downstream visualization layer, for example:
   - Top-10 feature importance bar chart
   - For classification: confusion matrix heatmap, ROC curve, PR curve
   - For regression: residual plot, predicted vs actual scatter plot, error distribution plot
   - Learning curve or validation curve

5. Technical recommendations:
   - Possible model improvement directions
   - Feature engineering opportunities
   - Useful future data collection ideas
   - Monitoring and model update strategy

6. Output language:
   - Write the entire report in {report_language_name}.
   - Do not mix languages.
   - If the requested language is English, every heading, paragraph, table label, and bullet point must be in English.

[Output format]
Return plain Markdown only.
Do not wrap the answer in a code block.
Start directly with `# Executive Summary`.
"""

    def _get_business_translation_prompt(self) -> str:
        return """You are a senior business analyst who is skilled at turning data science results into action plans that business teams can execute directly.

Translate the technical analysis into practical business recommendations based on the project context and JSON data.

[Project context]
{project_context}

[Technical report]
{technical_report}

[JSON data]
{json_data}

[Report requirements]

0. Topic adaptation:
   - Infer the business scenario strictly from the project context and JSON.
   - Do not assume this is an education, student, dropout, or advisor scenario unless the JSON clearly says so.
   - The names of people, teams, departments, and processes must match the current project theme.
   - If the JSON provides `industry`, `use_case`, `target_audience`, or `stakeholders`, use them.

1. Structure:
   - # Executive Summary
   - ## 1. Key Findings
   - ## 2. Immediate Action Recommendations
   - ## 3. ROI Analysis
   - ## 4. Risk Notes
   - ## 5. Implementation Roadmap

2. Writing style:
   - Written for non-technical managers
   - Avoid unnecessary technical jargon
   - Use business language that matches the project theme
   - Each recommendation should include owner, timeline, expected result, and required resources
   - Use concrete numbers whenever possible

3. Business translation examples:
   - Technical metric -> explain what it means for business decisions
   - Important feature -> explain why it should be monitored operationally
   - Recall gap -> explain what may still be missed and where manual review is needed
   - Regression error -> explain whether the prediction error is acceptable in the real use case

4. ROI analysis:
   - If the JSON contains cost, benefit, or conversion-value fields, present investment, expected return, net benefit, and ROI in a table.
   - If the JSON does not contain enough information, say that exact ROI cannot be calculated and list the missing inputs.

5. Recommendation format:
   - Priority
   - Action
   - Owner
   - Timeline
   - Expected result
   - Required resources

6. Risk notes:
   - Business implications of model limitations
   - Ethics, compliance, or privacy concerns
   - Implementation challenges such as data access, manual review cost, or user adoption

7. Output language:
   - Write the entire report in {report_language_name}.
   - Do not mix languages.
   - If the requested language is English, every heading, paragraph, table label, and bullet point must be in English.

[Output format]
Return plain Markdown only.
Do not wrap the answer in a code block.
Start directly with `# Executive Summary`.
"""

    def generate_reports(
        self,
        json_file_path: str,
        save_reports: bool = True,
    ) -> Dict[str, str]:
        json_data = self._load_json_file(json_file_path)
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        project_context = self._build_project_context(json_data)
        report_language_name = self._describe_report_language(
            self._resolve_report_language(json_data)
        )

        print("=" * 60)
        print("开始生成报告...")
        print("=" * 60)

        result = self.full_pipeline(
            {
                "json_data": json_str,
                "project_context": project_context,
                "report_language_name": report_language_name,
            }
        )
        technical_report = result["technical_report"]
        business_report = result["business_report"]

        if save_reports:
            self._save_reports(technical_report, business_report)

        return {
            "technical_report": technical_report,
            "business_report": business_report,
        }

    def generate_technical_report_only(
        self,
        json_file_path: str,
        save_report: bool = True,
    ) -> str:
        json_data = self._load_json_file(json_file_path)
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        project_context = self._build_project_context(json_data)
        report_language_name = self._describe_report_language(
            self._resolve_report_language(json_data)
        )
        result = self.technical_chain(
            {
                "json_data": json_str,
                "project_context": project_context,
                "report_language_name": report_language_name,
            }
        )
        technical_report = result["technical_report"]

        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"technical_report_{timestamp}.md"
            with open(path, "w", encoding="utf-8") as f:
                f.write(technical_report)
            print(f"✅ 技术报告已保存: {path}")

        return technical_report

    def generate_business_report_only(
        self,
        json_file_path: str,
        technical_report: Optional[str] = None,
        save_report: bool = True,
    ) -> str:
        json_data = self._load_json_file(json_file_path)
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        project_context = self._build_project_context(json_data)
        report_language_name = self._describe_report_language(
            self._resolve_report_language(json_data)
        )

        if technical_report is None:
            technical_report = self.generate_technical_report_only(
                json_file_path,
                save_report=False,
            )

        result = self.business_chain(
            {
                "json_data": json_str,
                "technical_report": technical_report,
                "project_context": project_context,
                "report_language_name": report_language_name,
            }
        )
        business_report = result["business_report"]

        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"business_report_{timestamp}.md"
            with open(path, "w", encoding="utf-8") as f:
                f.write(business_report)
            print(f"✅ 业务报告已保存: {path}")

        return business_report

    def _save_reports(self, technical_report: str, business_report: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        technical_path = self.output_dir / f"technical_report_{timestamp}.md"
        business_path = self.output_dir / f"business_report_{timestamp}.md"

        with open(technical_path, "w", encoding="utf-8") as f:
            f.write(technical_report)
        with open(business_path, "w", encoding="utf-8") as f:
            f.write(business_report)

        print(f"\n✅ 技术报告已保存: {technical_path}")
        print(f"✅ 业务报告已保存: {business_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="读取上游智能体输出的 JSON，并生成技术报告与业务报告"
    )
    parser.add_argument(
        "--json",
        type=str,
        default="example_pipeline_output.json",
        help="输入的JSON文件路径（通常由组员生成并交付）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="报告输出目录",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="使用的模型名称（默认优先读取 OPENAI_MODEL）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "technical", "business"],
        default="both",
        help="生成模式：both=两个报告，technical=仅技术报告，business=仅业务报告",
    )

    args = parser.parse_args()

    generator = MultiAgentReportGenerator(
        model_name=args.model,
        output_dir=args.output_dir,
    )

    if args.mode == "both":
        generator.generate_reports(args.json)
    elif args.mode == "technical":
        generator.generate_technical_report_only(args.json)
    else:
        generator.generate_business_report_only(args.json)

    print("\n" + "=" * 60)
    print("报告生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
