#!/bin/bash
# 多智能体报告生成系统 - 快速开始脚本
set -uo pipefail

VENV_DIR=".venv311"
PYTHON_BIN=""

if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "❌ 错误：未找到Python 3"
    echo "请安装Python 3.10~3.13（推荐3.11）"
    exit 1
fi

echo "=========================================="
echo "多智能体报告生成系统 - 快速开始"
echo "=========================================="
echo ""

# 检查Python版本
echo "1. 检查Python版本..."
"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit("❌ 当前Python版本不兼容，请使用3.10~3.13（推荐3.11）")
PY
echo "✅ Python版本检查通过"
echo ""

# 创建虚拟环境
echo "2. 准备虚拟环境..."
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
echo "✅ 虚拟环境已就绪: $VENV_DIR"
echo ""

# 安装依赖
echo "3. 安装依赖包..."
"$VENV_PIP" install --upgrade pip
"$VENV_PIP" install -r requirements_langchain.txt
echo "✅ 依赖安装完成"
echo ""

# 检查.env文件
echo "4. 检查配置文件..."
if [ ! -f .env ]; then
    echo "⚠️  未找到.env文件，正在创建..."
    cp .env.example .env
    echo "📝 请编辑.env文件，填入OpenAI API密钥："
    echo "   OPENAI_API_KEY=your_openai_api_key_here"
    echo ""
    echo "按Enter键继续（确保已配置API密钥）..."
    read
fi
echo "✅ 配置文件检查完成"
echo ""

# 创建输出目录
echo "5. 创建输出目录..."
mkdir -p reports
echo "✅ 输出目录已创建: reports/"
echo ""

# 运行示例
echo "6. 运行示例..."
echo "使用结构化 JSON 生成报告..."
"$VENV_PY" multi_agent_report_generator.py --json example_pipeline_output.json

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 报告生成成功！"
    echo "=========================================="
    echo ""
    echo "查看生成的报告："
    echo "  - 技术报告: reports/technical_report_*.md"
    echo "  - 业务报告: reports/business_report_*.md"
    echo ""
    echo "下一步："
    echo "  1. 查看reports/目录中的报告"
    echo "  2. 阅读README_MULTI_AGENT_SYSTEM.md了解更多用法"
    echo "  3. 使用组员交付的 JSON：$VENV_PY multi_agent_report_generator.py --json your_team_output.json"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 报告生成失败"
    echo "=========================================="
    echo ""
    echo "可能的原因："
    echo "  1. OpenAI API密钥未配置或无效"
    echo "  2. 输入JSON缺少必需字段"
    echo "  3. 网络连接问题"
    echo "  4. API配额不足"
    echo ""
    echo "请检查.env文件中的OPENAI_API_KEY配置"
    echo "详细错误信息请查看上方输出"
    echo ""
fi
