#!/usr/bin/env python3
"""
测试脚本 - 验证 JSON 驱动的报告生成系统
不需要 OpenAI API，仅测试输入结构和本地文件。
"""

import json
from pathlib import Path


def test_json_structure():
    """测试上游智能体交付的 JSON 结构"""
    print("=" * 60)
    print("测试1: 上游JSON结构验证")
    print("=" * 60)
    
    json_file = "example_pipeline_output.json"
    
    if not Path(json_file).exists():
        print(f"❌ 错误：未找到{json_file}")
        return False
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查必需字段
    required_sections = [
        "meta",
        "data_understanding",
        "data_cleaning",
        "feature_engineering",
        "modeling",
        "evaluation",
        "business_context"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in data:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ 缺少必需字段: {', '.join(missing_sections)}")
        return False
    
    print("✅ JSON结构验证通过")
    print(f"   - 可直接作为上游交付输入: {json_file}")
    print(f"   - 包含所有必需字段: {', '.join(required_sections)}")
    
    # 显示关键信息
    print(f"\n关键信息:")
    print(f"   - 数据集: {data['meta']['dataset_name']}")
    print(f"   - 样本数: {data['data_understanding']['n_rows']}")
    print(f"   - 特征数: {data['data_understanding']['n_cols']}")
    print(f"   - 最佳模型: {data['modeling']['best_model']['name']}")
    print(f"   - F1分数: {data['evaluation']['metrics']['f1_score']}")
    print(f"   - 准确率: {data['evaluation']['metrics']['accuracy']}")
    
    return True


def test_team_template_structure():
    """测试给组员的新模板结构"""
    print("\n" + "=" * 60)
    print("测试2: 团队JSON模板验证")
    print("=" * 60)

    json_file = "team_json_template.json"

    if not Path(json_file).exists():
        print(f"❌ 错误：未找到{json_file}")
        return False

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_sections = [
        "project_info",
        "dataset_summary",
        "pipeline_trace",
        "model_results",
        "interpretability",
        "risk_scoring",
        "business_constraints",
    ]

    missing_sections = [section for section in required_sections if section not in data]
    if missing_sections:
        print(f"❌ 新模板缺少字段: {', '.join(missing_sections)}")
        return False

    print("✅ 团队JSON模板验证通过")
    print(f"   - 可直接发给组员作为标准接口: {json_file}")
    print(f"   - 包含新版结构字段: {', '.join(required_sections)}")
    return True


def test_file_structure():
    """测试文件结构"""
    print("\n" + "=" * 60)
    print("测试3: 文件结构验证")
    print("=" * 60)
    
    required_files = [
        "README.md",
        "multi_agent_report_generator.py",
        "example_pipeline_output.json",
        "team_json_template.json",
        "requirements_langchain.txt",
        ".env.example",
        "README_MULTI_AGENT_SYSTEM.md",
        "quick_start.sh"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 文件结构验证通过")
    print(f"   - 所有必需文件都存在")
    
    return True


def test_code_imports():
    """测试代码导入"""
    print("\n" + "=" * 60)
    print("测试4: 代码导入验证")
    print("=" * 60)
    
    try:
        # 测试基本导入
        import json
        import os
        from pathlib import Path
        from datetime import datetime
        
        print("✅ 基本库导入成功")
        
        # 测试LangChain导入（可能失败，因为可能未安装）
        try:
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            print("✅ LangChain库导入成功")
        except ImportError:
            print("⚠️  LangChain未安装（运行前需要: pip install -r requirements_langchain.txt）")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_prompt_templates():
    """测试核心方法存在"""
    print("\n" + "=" * 60)
    print("测试5: 核心方法验证")
    print("=" * 60)
    
    try:
        # 读取主文件
        with open("multi_agent_report_generator.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键方法
        required_methods = [
            "validate_input_json",
            "_normalize_input_json",
            "_normalize_new_schema",
            "_get_technical_report_prompt",
            "_get_business_translation_prompt",
            "generate_reports",
            "generate_technical_report_only",
            "generate_business_report_only"
        ]
        
        missing_methods = []
        for method in required_methods:
            if method not in content:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ 缺少方法: {', '.join(missing_methods)}")
            return False
        
        print("✅ 核心方法验证通过")
        print(f"   - 所有必需方法都存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("多智能体报告生成系统 - JSON交付模式测试")
    print("=" * 60)
    print()
    
    tests = [
        ("旧版JSON结构", test_json_structure),
        ("团队模板结构", test_team_template_structure),
        ("文件结构", test_file_structure),
        ("代码导入", test_code_imports),
        ("Prompt模板", test_prompt_templates)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}测试异常: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统已准备好接收组员提供的 JSON。")
        print("\n下一步:")
        print("  1. 配置.env文件（添加OPENAI_API_KEY）")
        print("  2. 安装依赖: pip install -r requirements_langchain.txt")
        print("  3. 让组员交付标准 JSON")
        print("  4. 运行系统: python3 multi_agent_report_generator.py --json your_team_output.json")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
