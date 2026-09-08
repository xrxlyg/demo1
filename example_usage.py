"""
example_usage.py  (v6 适配版)

"""

import json
import math
import os
from datetime import datetime

from llm_wrapper import LLMWrapper
from interview_orchestrator import InterviewOrchestrator
from agents.candidate_simulator_agent import CandidateSimulatorAgent


# ── 格式化工具 ─────────────────────────────────────────────────────────────────

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def print_evaluation(eval_result):
    if not eval_result:
        return
    print("\n[评估结果]")
    print(f"评估类型: {eval_result.get('评估类型', 'N/A')}")
    print(f"考察目标: {eval_result.get('考察目标', 'N/A')}")
    print(f"综合得分: {eval_result.get('综合得分', 0.0):.2f}")

    print("\n各维度得分:")
    skip = {"评估类型", "考察目标", "综合得分", "评语", "不足之处", "建议追问方向"}
    for key, value in eval_result.items():
        if key not in skip and isinstance(value, (int, float)):
            print(f"  - {key}: {value:.2f}")

    if "评语" in eval_result:
        print(f"\n评语: {eval_result['评语']}")
    if "不足之处" in eval_result:
        print(f"不足: {eval_result['不足之处']}")
    if "建议追问方向" in eval_result:
        print(f"建议: {eval_result['建议追问方向']}")


# ── 序列化工具 ─────────────────────────────────────────────────────────────────

def safe_serialize(obj):
    """
    递归序列化，处理 float NaN/Infinity、set/frozenset、不可序列化对象。
    """
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(i) for i in obj]
    if isinstance(obj, set):
        return [safe_serialize(i) for i in sorted(obj, key=str)]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def extract_dialogue_history(orchestrator) -> list:
    """
    InterviewState.add_dialogue 以字典形式存储，逐项安全提取。
    """
    result = []
    for item in orchestrator.state.dialogue_history:
        try:
            if isinstance(item, dict):
                result.append({
                    "question":      item.get("question", ""),
                    "answer":        item.get("answer", ""),
                    "skill":         item.get("skill", ""),
                    "question_type": item.get("question_type", ""),
                    "difficulty":    item.get("difficulty", ""),
                    "is_followup":   item.get("is_followup", False),
                })
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                result.append({"question": item[0], "answer": item[1]})
            else:
                result.append({"raw": str(item)})
        except Exception as e:
            result.append({"error": str(e), "raw": str(item)})
    return result


# ── SkillGraph 工具 ────────────────────────────────────────────────────────────

def extract_skill_tree(orchestrator) -> dict:
    """
    从 orchestrator._skill_graph 提取完整树结构，用于 JSON 保存。

    SkillGraph.summary() 返回结构：
    {
        "total_nodes":     int,
        "root_count":      int,
        "coverage_rate":   float,
        "uncovered_roots": [str, ...],
        "nodes": {
            "节点名": {
                "depth":        int,
                "weight":       float,
                "observations": int,
                "children":     [str, ...]   <- 子节点名称列表
            },
            ...
        }
    }
    """
    skill_graph = getattr(orchestrator, "_skill_graph", None)
    if skill_graph is None:
        return {}
    return safe_serialize(skill_graph.summary())


def print_skill_tree(orchestrator):
    """
    按层级递归打印 SkillGraph 树结构。
    每个节点显示 depth / weight / observations。

    依赖 skill_graph.nodes（SkillNode 对象），
    通过 node.children（子节点名称列表）递归渲染。
    """
    skill_graph = getattr(orchestrator, "_skill_graph", None)
    if skill_graph is None:
        print("  （无 SkillGraph，使用平铺技能列表）")
        return

    nodes = skill_graph.nodes
    if not nodes:
        print("  （技能树为空）")
        return

    def _print_node(name: str, indent: int = 0):
        node = nodes.get(name)
        if node is None:
            return
        prefix  = "  " * indent + ("└─ " if indent > 0 else "")
        obs_str = f"  [考察 {node.observations} 次]" if node.observations > 0 else ""
        print(f"{prefix}{name}  (depth={node.depth}, w={node.weight:.1f}){obs_str}")
        for child_name in node.children:
            _print_node(child_name, indent + 1)

    roots = skill_graph.roots()
    for root in roots:
        _print_node(root.name)

    max_depth = max(n.depth for n in nodes.values())
    print(
        f"\n  共 {len(nodes)} 个节点，"
        f"根节点 {len(roots)} 个，"
        f"最大深度 {max_depth}，"
        f"覆盖率 {skill_graph.coverage_rate() * 100:.0f}%"
    )


def print_skill_tree_stats(orchestrator):
    """
    面试结束后打印节点考察情况，按 已考察 / 未考察 分组，组内按深度排序。
    """
    skill_graph = getattr(orchestrator, "_skill_graph", None)
    if skill_graph is None:
        return

    nodes        = skill_graph.nodes
    assessed     = [(name, n) for name, n in nodes.items() if n.observations > 0]
    not_assessed = [(name, n) for name, n in nodes.items() if n.observations == 0]

    if assessed:
        print(f"  已考察节点（{len(assessed)} 个）:")
        for name, node in sorted(assessed, key=lambda x: (x[1].depth, x[0])):
            indent = "  " * (node.depth + 1)
            print(f"{indent}{name}  (depth={node.depth}, 考察 {node.observations} 次)")

    if not_assessed:
        print(f"\n  未考察节点（{len(not_assessed)} 个）:")
        for name, node in sorted(not_assessed, key=lambda x: (x[1].depth, x[0])):
            indent = "  " * (node.depth + 1)
            print(f"{indent}{name}  (depth={node.depth})")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def run_interview_simulation():

    skill_pool = [
        "java", "kotlin", "python", "c++", "c#", "dart", "go", "rust",
        "typescript", "javascript", "swift", "objective-c", "node.js",
        "android", "ios", "flutter", "react native", "android studio", "xcode",
        "html", "css", "react", "vue", "angular", "webpack", "前端框架",
        "spring", "spring boot", "分布式系统", "数据库", "redis",
        "消息队列", "restful api", "graphql",
        "git", "jenkins", "maven", "gradle", "docker", "kubernetes",
        "ci/cd", "jira", "confluence",
        "aws", "azure", "gcp", "云计算", "devops",
        "多线程", "内存管理", "网络编程", "数据结构", "算法", "设计模式",
        "性能优化", "系统架构", "安全",
        "机器学习", "深度学习", "自然语言处理", "计算机视觉", "大数据",
        "数据分析", "tensorflow", "pytorch",
        "车载系统", "智能语音", "物联网", "嵌入式", "自动驾驶", "车联网",
    ]

    dimension_pool = [
        "人际互动与沟通能力",
        "内在动机与驱动力",
        "性格特质",
        "执行力与结果导向",
        "文化契合度与价值观",
        "认知与解决问题能力",
    ]

    dimension_tree = {
        "人际互动与沟通能力": [
            "沟通清晰度与倾听能力", "同理心与关系建立", "影响力与说服力",
            "冲突管理与协商能力", "跨文化沟通与协作",
        ],
        "内在动机与驱动力": [
            "成就导向", "求知欲与学习热情", "自主性与主人翁精神",
            "目标感与自我规划", "抗挫折动力",
        ],
        "性格特质": [
            "情绪稳定性与抗压能力", "尽责性与可靠性", "外向性与协作倾向",
            "自信与积极心态", "开放性与包容性",
        ],
        "执行力与结果导向": [
            "计划组织与优先级管理", "行动力与克服障碍", "结果意识与闭环能力",
            "自我管理与自律性", "目标达成与责任追踪",
        ],
        "文化契合度与价值观": [
            "核心价值观认同", "工作风格与团队契合", "诚信与职业道德",
            "协作共赢_vs._个人英雄主义", "文化增值与多元融合",
        ],
        "认知与解决问题能力": [
            "逻辑思维与分析能力", "创新思维与灵活性", "批判性思维与决策判断",
            "学习能力与知识迁移", "复杂问题解决能力",
        ],
    }

    job_profile = {
        "job_id":   "J20241101",
        "job_name": "后端开发工程师",
        "raw_text": (
            "岗位职责：1. 负责公司核心业务系统的后端开发和维护。"
            "2. 参与系统架构设计，优化系统性能。"
            "3. 与前端、产品团队协作，推动项目落地。"
            "岗位要求：1. 熟练掌握Java/Python等编程语言，熟悉Spring框架。"
            "2. 了解分布式系统、数据库、缓存等技术。"
            "3. 良好的沟通能力和团队协作精神。"
        ),
        "recruit_type": "社招",
    }

    candidate_resume = """张三
    电话： +86 138-xxxx-xxxx | 邮箱： zhangsan@email.com | 所在地： 深圳

    个人简介
    专注于后端开发领域的软件工程师，拥有扎实的Java技术栈和分布式系统开发经验。

    教育背景
    华南理工大学 | 计算机科学与技术 | 工学学士
    2019年9月 - 2023年6月

    工作/实习经历
    ABC科技有限公司 | 后端开发工程师（实习）
    2022年7月 - 2022年10月 | 上海

    参与电商平台订单系统重构：负责重构优惠券计算模块，通过引入策略模式，
    将代码耦合度降低40%，提升了代码的可维护性和扩展性。

    性能优化：通过分析和优化慢查询SQL，并使用Redis缓存热点商品信息，
    将订单查询接口的平均响应时间从350ms降低至80ms。
    """

    candidate_name = "张三"

    # ── 初始化 ────────────────────────────────────────────────────────────────
    print_section("面试系统初始化 (v6)")
    print("核心架构：")
    print("  Stage 1  JobAnalystAgent      — JD原文 → 岗位分析")
    print("  Stage 2  SkillTreeAgent       — 两阶段建树（宽树 + 逐节点深扩）")
    print("  Stage 3  DimensionExpertAgent — 维度 → 子维度映射")
    print("  Stage 4  AbilityModel + SkillGraph 初始化")
    print()
    print("选题策略：score(q) = α·IG(s) + β·CG(s) + γ·DM(s) + δ·DR(s)")

    llm = LLMWrapper(model_name="qwen-plus")

    # v6 构造：去掉 min_skills / max_skills
    orchestrator = InterviewOrchestrator(
        llm_wrapper=llm,
        skill_pool=skill_pool,
        dimension_pool=dimension_pool,
        dimension_tree=dimension_tree,
        # min_dimensions=1,
        # max_dimensions=2,
        save_training_data=True,
        training_data_dir="training_data",
        candidate_name=candidate_name,
        max_questions=6,
        # max_questions=8,
    )

    candidate_simulator = CandidateSimulatorAgent(llm)

    # ── 四阶段初始化 ──────────────────────────────────────────────────────────
    orchestrator.initialize_interview(job_profile)

    print(f"岗位:     {orchestrator.job_name}")
    print(f"识别技能: {orchestrator.required_skills}")
    print(
        f"评估维度 ({len(orchestrator.required_dimensions)}): "
        f"{', '.join([d['dimension'] for d in orchestrator.required_dimensions])}"
    )

    # ── 打印建好的技能树（面试前）────────────────────────────────────────────
    print_section("技能树结构（SkillTreeAgent 两阶段建树结果）")
    print_skill_tree(orchestrator)

    # ── 面试主循环 ────────────────────────────────────────────────────────────
    print_section("面试开始")

    question = orchestrator.generate_opening_question()
    print(f"[面试官] {question}\n")

    for round_num in range(1, 9):
        print(f"\n{'—' * 40}")
        print(f"第 {round_num} 轮")
        print(f"{'—' * 40}")

        answer = candidate_simulator.run(question, candidate_resume)
        print(f"\n[候选人] {answer}\n")

        next_question, evaluation = orchestrator.process_response(answer, question)

        if evaluation:
            print_evaluation(evaluation)

        # 选题评分明细（调试用，可注释掉）— 移到 process_response 之后
        debug_scores = orchestrator.get_selector_debug()
        if debug_scores:
            print("\n[选题评分明细 Top 3]")
            for item in debug_scores[:3]:
                print(
                    f"  {item['skill']:25s}  "
                    f"total={item['total']:.3f}  "
                    f"IG={item['IG']:.3f}  "
                    f"CG={item['CG']:.3f}  "
                    f"DM={item['DM']:.3f}  "
                    f"DR={item['DR']:.3f}"
                )

        print(f"\n[面试官] {next_question}\n")
        question = next_question

        if "问题吗" in question:
            print("\n[面试结束]")
            break

    # ── 整体评估 ──────────────────────────────────────────────────────────────
    print_section("整体评估报告")
    overall_eval = orchestrator.generate_overall_evaluation()

    # ability_model / skill_graph 单独打印，避免正文过长
    print_fields = {
        k: v for k, v in overall_eval.items()
        if k not in ("ability_model", "skill_graph")
    }
    print(json.dumps(print_fields, ensure_ascii=False, indent=2))

    # 打印各技能的 (μ, σ, n) 估计
    ability_model = overall_eval.get("ability_model", {})
    assessed_skills = {
        s: e for s, e in ability_model.get("skills", {}).items()
        if e.get("observations", 0) > 0
    }
    if assessed_skills:
        print("\n[AbilityModel — 技能能力估计]")
        for skill, est in sorted(assessed_skills.items()):
            print(
                f"  {skill:25s}  "
                f"μ={est['mean']:.3f}  "
                f"σ={est['std']:.3f}  "
                f"n={est['observations']}"
            )

    # ── 统计与覆盖率 ──────────────────────────────────────────────────────────
    print_section("面试统计与覆盖分析")
    summary = orchestrator.get_interview_summary()
    print(f"总问题数:     {summary['总问题数']}")
    print(f"讨论经历数:   {summary['讨论经历数']}")
    print(f"覆盖技能数:   {summary['覆盖技能数']}")
    print(f"评估维度数:   {summary['评估维度数']}")
    print(f"最终难度级别: {summary['当前难度']}")
    print(f"最终阶段:     {summary['最终阶段']}")
    print(f"\n技能覆盖率:   {summary['技能覆盖率']}")
    print(f"维度覆盖率:   {summary['维度覆盖率']}")

    # ── 面试后技能树 ──────────────────────────────────────────────────────────
    print_section("技能树考察情况（面试后）")
    print_skill_tree(orchestrator)
    print()
    print_skill_tree_stats(orchestrator)

    # print(f"\n困惑检测次数: {len(orchestrator.confusion_history)}")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    print_section("保存面试结果")

    output_dir = "interview_results"
    os.makedirs(output_dir, exist_ok=True)

    date_str        = datetime.now().strftime("%Y%m%d")
    result_filename = os.path.join(
        output_dir,
        f"{date_str}_{candidate_name}_{orchestrator.job_name}.json",
    )

    # extract_skill_tree 使用 SkillGraph.summary() 标准结构：
    # {total_nodes, root_count, coverage_rate, uncovered_roots,
    #  nodes: {name: {depth, weight, observations, children: [str, ...]}}}
    skill_tree_data = extract_skill_tree(orchestrator)

    result_data = {
        "session_metadata": {
            "candidate_name":      candidate_name,
            "job_name":            orchestrator.job_name,
            "interview_date":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "required_skills":     orchestrator.required_skills,
            "required_dimensions": [
                d["dimension"] for d in orchestrator.required_dimensions
            ],
        },
        "skill_tree_raw":       safe_serialize(orchestrator.skill_hierarchy),
        "skill_tree":           skill_tree_data,
        "overall_evaluation":   safe_serialize(overall_eval),
        "interview_summary":    safe_serialize(summary),
        "dialogue_history":     extract_dialogue_history(orchestrator),
        "answer_evaluations":   safe_serialize(orchestrator.answer_evaluations),
        "coverage_analysis": {
            "skill_coverage":     safe_serialize(orchestrator.skill_coverage),
            "dimension_coverage": safe_serialize(orchestrator.dimension_coverage),
        },
    }

    try:
        with open(result_filename, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        file_size = os.path.getsize(result_filename) / 1024
        print(f"面试结果已保存至: {result_filename}")
        print(f"文件大小:         {file_size:.2f} KB")
        print(f"技能树节点数:     {len(skill_tree_data.get('nodes', {}))}")
        print(f"技能树根节点:     {skill_tree_data.get('root_count', 0)} 个")
        print(f"技能树覆盖率:     {skill_tree_data.get('coverage_rate', 0) * 100:.0f}%")
    except Exception as e:
        print(f"保存失败，错误原因: {e}")
        print("逐字段检查序列化情况：")
        for key, value in result_data.items():
            try:
                json.dumps(value, ensure_ascii=False)
                print(f"  [正常] {key}")
            except Exception as field_err:
                print(f"  [失败] {key}: {field_err}")


if __name__ == "__main__":
    print("=" * 80)
    print("智能面试系统演示 (v6)".center(80))
    print("=" * 80)
    run_interview_simulation()