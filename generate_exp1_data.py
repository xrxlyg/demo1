"""
generate_exp1_data.py

实验一数据生成器
================
生成真实可用于论文的 annotated_samples，逻辑链：

    LLM 扮演候选人 → 给出回答
         ↓
    LLM 作为评估器 → 给出 evaluator_score（视为"人工评分"代理）
         ↓
    AbilityModel.update_skill() → 给出 ability_mean_after
         ↓
    写入 exp1_data.json，供 run_exp1() 直接消费

为什么这样做不虚高：
    - 候选人回答由 LLM 基于"真实能力档"生成，文本内容真实
    - evaluator_score 由独立 LLM 调用打分，与 AbilityModel 无耦合
    - AbilityModel 是在 evaluator_score 输入后再更新的，不是直接从真实能力算的
    - 两者之间存在真实的"语言生成 + 语言理解"误差，相关性反映真实信噪比

用法：
    export ANTHROPIC_API_KEY=sk-...   # 或通过 --api-key 传入
    python generate_exp1_data.py --candidates 20 --output exp1_data.json

    # 指定技能 / 题数 / 每题问题数
    python generate_exp1_data.py --candidates 10 --skills 分布式系统 数据库 算法 \\
        --questions-per-skill 3 --output exp1_data.json

    # 已有数据不够，追加
    python generate_exp1_data.py --candidates 10 --append --output exp1_data.json
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

import requests

# ── 本地模块 ──────────────────────────────────────────────────────────────────
try:
    from ability_model import AbilityModel
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from ability_model import AbilityModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 候选人档
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateProfile:
    """一名候选人的完整设定。"""
    candidate_id:  int
    level:         str          # junior / mid / senior
    true_theta:    float        # 真实综合能力 1-10（用于事后分析，不传给 LLM）
    skill_bias:    Dict[str, float]  # 技能偏差：θ_skill = true_theta + bias[skill]
    persona:       str          # LLM 角色扮演指令片段


# 三类档位配置
LEVEL_CONFIG = {
    "junior": {
        "theta_range": (2.0, 4.5),
        "persona": (
            "你是一名刚毕业1-2年的初级后端工程师，基础扎实但缺乏大规模生产经验。"
            "回答时会展示对概念的基本理解，但对细节和边界情况不够熟悉，"
            "偶尔会给出不完整或略有偏差的答案。"
        ),
    },
    "mid": {
        "theta_range": (5.0, 7.5),
        "persona": (
            "你是一名有3-5年经验的中级后端工程师，有实际项目经验。"
            "能够清晰解释概念并结合实际场景，偶尔对高级特性或边界情况不够深入。"
        ),
    },
    "senior": {
        "theta_range": (7.5, 9.5),
        "persona": (
            "你是一名有7年以上经验的高级后端工程师，深入理解底层原理。"
            "回答系统性强，能主动讨论权衡取舍、故障案例和优化思路，"
            "展示出对生产环境复杂性的深刻理解。"
        ),
    },
}

DEFAULT_SKILLS = [
    "分布式系统",
    "数据库索引与查询优化",
    "Redis缓存设计",
    "消息队列",
    "系统设计",
]

SKILL_QUESTIONS: Dict[str, List[str]] = {
    "分布式系统": [
        "请解释 CAP 定理，并举例说明在实际系统中如何做出取舍。",
        "什么是分布式事务？你在项目中是如何处理分布式事务一致性问题的？",
        "谈谈你对 Raft 共识算法的理解，它如何保证日志一致性？",
    ],
    "数据库索引与查询优化": [
        "请解释 B+ 树索引的结构，以及为什么数据库通常选择它而不是哈希索引？",
        "什么是覆盖索引？能举一个用覆盖索引避免回表的例子吗？",
        "遇到慢查询时你的排查思路是什么？",
    ],
    "Redis缓存设计": [
        "Redis 的持久化机制 RDB 和 AOF 有什么区别？各自适合什么场景？",
        "如何处理缓存穿透、缓存击穿和缓存雪崩？",
        "Redis 集群模式下的数据分片是如何实现的？",
    ],
    "消息队列": [
        "Kafka 如何保证消息不丢失？消费者端需要注意哪些问题？",
        "什么是消息幂等性？你是如何在消费者侧实现幂等处理的？",
        "Kafka 的 partition 和 consumer group 的关系是什么？",
    ],
    "系统设计": [
        "请设计一个支持 10 万 QPS 的短链接服务，描述主要组件和数据流。",
        "如何设计一个分布式限流系统？",
        "设计一个高可用的秒杀系统，说明如何防止超卖。",
    ],
}


def generate_candidate_profile(
    candidate_id: int,
    level: str,
    skills: List[str],
    rng: random.Random,
) -> CandidateProfile:
    cfg = LEVEL_CONFIG[level]
    lo, hi = cfg["theta_range"]
    true_theta = rng.uniform(lo, hi)
    # 每个技能有 ±2 分的随机偏差，模拟技能不均
    skill_bias = {s: rng.gauss(0, 1.2) for s in skills}
    return CandidateProfile(
        candidate_id=candidate_id,
        level=level,
        true_theta=true_theta,
        skill_bias=skill_bias,
        persona=cfg["persona"],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Anthropic API 封装
# ═════════════════════════════════════════════════════════════════════════════

class ClaudeClient:
    """最小化 Anthropic claude-sonnet-4-20250514 调用封装（无第三方 SDK 依赖）。"""

    MODEL   = "claude-sonnet-4-20250514"
    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, retry: int = 3, retry_delay: float = 2.0):
        self.headers = {
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        self.retry       = retry
        self.retry_delay = retry_delay

    def chat(
        self,
        system:     str,
        user:       str,
        max_tokens: int = 800,
        temperature: float = 0.7,
    ) -> str:
        payload = {
            "model":       self.MODEL,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "system":      system,
            "messages":    [{"role": "user", "content": user}],
        }
        for attempt in range(1, self.retry + 1):
            try:
                resp = requests.post(
                    self.API_URL, headers=self.headers,
                    json=payload, timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["content"][0]["text"].strip()
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    wait = self.retry_delay * (2 ** attempt)
                    log.warning("Rate limit，等待 %.1fs (attempt %d/%d)", wait, attempt, self.retry)
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt == self.retry:
                    raise
                log.warning("API 错误 %s，重试 %d/%d", e, attempt, self.retry)
                time.sleep(self.retry_delay)
        raise RuntimeError("API 调用失败，已达最大重试次数")


# ═════════════════════════════════════════════════════════════════════════════
# 候选人回答生成
# ═════════════════════════════════════════════════════════════════════════════

ANSWER_SYSTEM = """\
{persona}

以下是面试官提出的技术问题，请用中文作答。
要求：
- 语气自然，像真实面试口语表达
- 回答长度 120-250 字
- 不要使用 Markdown 格式，直接输出纯文本
- 不要在回答开头加"好的"/"当然"等口头语
"""

def generate_answer(client: ClaudeClient, profile: CandidateProfile, question: str) -> str:
    system = ANSWER_SYSTEM.format(persona=profile.persona)
    return client.chat(system=system, user=question, max_tokens=400, temperature=0.85)


# ═════════════════════════════════════════════════════════════════════════════
# LLM 评分
# ═════════════════════════════════════════════════════════════════════════════

EVAL_SYSTEM = """\
你是一名严格、专业的技术面试评估专家。
请对候选人的回答进行评分，输出 **仅一个 JSON 对象**，不要输出任何其他内容（包括 Markdown 代码块）。

评分维度：
1. 准确性（accuracy）：概念和技术细节是否正确，0-10 分
2. 深度（depth）：是否涉及底层原理和边界情况，0-10 分
3. 实践性（practicality）：是否结合真实场景和经验，0-10 分
4. 表达清晰度（clarity）：逻辑是否清晰，表达是否准确，0-10 分

综合得分（overall）= 准确性 * 0.35 + 深度 * 0.30 + 实践性 * 0.20 + 清晰度 * 0.15
综合得分保留两位小数，区间 0-10。

输出格式（严格 JSON，无注释）：
{
  "accuracy":     <float>,
  "depth":        <float>,
  "practicality": <float>,
  "clarity":      <float>,
  "overall":      <float>,
  "comment":      "<一句话评语，不超过50字>"
}
"""

EVAL_USER = """\
【技术问题】
{question}

【候选人回答】
{answer}
"""

def evaluate_answer(client: ClaudeClient, question: str, answer: str) -> Dict[str, Any]:
    """调用 LLM 评分，返回包含 overall 字段的字典。"""
    raw = client.chat(
        system=EVAL_SYSTEM,
        user=EVAL_USER.format(question=question, answer=answer),
        max_tokens=300,
        temperature=0.2,   # 评分要稳定
    )
    # 清理可能的 markdown 代码块
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        log.error("评分 JSON 解析失败，原始输出：\n%s", raw)
        # fallback：返回 5.0 中性分
        return {"accuracy": 5.0, "depth": 5.0, "practicality": 5.0,
                "clarity": 5.0, "overall": 5.0, "comment": "解析失败"}


# ═════════════════════════════════════════════════════════════════════════════
# 核心数据生成逻辑
# ═════════════════════════════════════════════════════════════════════════════

def generate_exp1_data(
    client:               ClaudeClient,
    n_candidates:         int       = 20,
    skills:               Optional[List[str]] = None,
    questions_per_skill:  int       = 3,
    seed:                 int       = 42,
    progress_file:        Optional[str] = None,  # 断点续传
) -> List[Dict[str, Any]]:
    """
    主生成函数。

    Returns:
        annotated_samples 列表，每条：
        {
            candidate_id, level, skill, question_index,
            question, answer,
            evaluator_score:   float,  # LLM 评分（视为人工评分代理）
            ability_mean_after: float, # AbilityModel.mean * 10（更新后）
            true_theta:        float,  # 候选人真实能力（仅用于事后分析）
            eval_detail:       dict,   # 四维评分 + 评语
        }
    """
    rng = random.Random(seed)
    if skills is None:
        skills = DEFAULT_SKILLS

    # ── 断点续传：加载已有进度 ──────────────────────────────────────────────
    completed_keys: set = set()
    samples: List[Dict] = []
    if progress_file and Path(progress_file).exists():
        with open(progress_file, encoding="utf-8") as f:
            samples = json.load(f)
        completed_keys = {
            (s["candidate_id"], s["skill"], s["question_index"])
            for s in samples
        }
        log.info("断点续传：已加载 %d 条样本", len(samples))

    # ── 均匀分配档位：junior / mid / senior 各占 1/3 ──────────────────────
    levels = (["junior"] * (n_candidates // 3)
              + ["mid"]    * (n_candidates // 3)
              + ["senior"] * (n_candidates - 2 * (n_candidates // 3)))
    rng.shuffle(levels)

    total_expected = n_candidates * len(skills) * questions_per_skill
    generated = len(samples)

    for cand_idx in range(n_candidates):
        profile = generate_candidate_profile(cand_idx, levels[cand_idx], skills, rng)
        ability_model = AbilityModel()

        # 已有进度：重放 AbilityModel 状态
        prior_scores = [
            s["evaluator_score"] for s in samples
            if s["candidate_id"] == cand_idx
        ]
        # 按记录顺序重放（粗略：只按 skill 顺序，精确版可按 question_index 排序）
        skill_prior = {sk: [] for sk in skills}
        for s in samples:
            if s["candidate_id"] == cand_idx:
                skill_prior[s["skill"]].append(s["evaluator_score"])
        for sk in skills:
            for sc in skill_prior[sk]:
                ability_model.update_skill(sk, sc)

        log.info(
            "候选人 %d/%d  level=%-6s  θ=%.2f",
            cand_idx + 1, n_candidates, profile.level, profile.true_theta,
        )

        for skill in skills:
            # 从题库 / 默认题库 取题
            q_pool = SKILL_QUESTIONS.get(skill, [
                f"请描述一下你对 {skill} 的理解，以及在实际项目中的应用经验。",
                f"在使用 {skill} 时你遇到过哪些挑战，是如何解决的？",
                f"请谈谈 {skill} 的核心原理和主要优缺点。",
            ])
            questions = (q_pool * ((questions_per_skill // len(q_pool)) + 1))[:questions_per_skill]

            for q_idx, question in enumerate(questions):
                key = (cand_idx, skill, q_idx)
                if key in completed_keys:
                    # 跳过已完成的，但要保持 ability_model 状态已在上方重放
                    continue

                log.info(
                    "  [%d/%d] skill=%-20s q=%d  正在生成回答...",
                    generated + 1, total_expected, skill, q_idx,
                )

                # Step 1: 生成回答
                answer = generate_answer(client, profile, question)

                # Step 2: 独立评分（evaluator_score 视为 human_score 代理）
                eval_result = evaluate_answer(client, question, answer)
                evaluator_score = float(eval_result.get("overall", 5.0))

                # Step 3: AbilityModel 更新
                ability_model.update_skill(skill, evaluator_score)
                ability_mean_after = ability_model.skills[skill].mean * 10.0

                record = {
                    "candidate_id":       cand_idx,
                    "level":              profile.level,
                    "skill":              skill,
                    "question_index":     q_idx,
                    "question":           question,
                    "answer":             answer,
                    "evaluator_score":    round(evaluator_score, 4),
                    "ability_mean_after": round(ability_mean_after, 4),
                    "true_theta":         round(profile.true_theta, 4),
                    "eval_detail":        eval_result,
                    # run_exp1 所需的字段别名
                    "human_score":        round(evaluator_score, 4),
                }
                samples.append(record)
                generated += 1

                # 断点续传：每生成一条立即写盘
                if progress_file:
                    _save_json(samples, progress_file)

                # 礼貌地控制 API 速率
                time.sleep(0.3)

    return samples


# ═════════════════════════════════════════════════════════════════════════════
# 数据质量报告
# ═════════════════════════════════════════════════════════════════════════════

def print_data_report(samples: List[Dict[str, Any]]) -> None:
    import numpy as np

    print("\n" + "=" * 60)
    print("  实验一数据质量报告")
    print("=" * 60)
    print(f"  总样本数：{len(samples)}")

    # 档位分布
    from collections import Counter
    level_dist = Counter(s["level"] for s in samples)
    print(f"  档位分布：{dict(level_dist)}")

    # 评分分布
    scores = np.array([s["evaluator_score"] for s in samples])
    print(f"  评分统计：mean={scores.mean():.2f}  std={scores.std():.2f}"
          f"  min={scores.min():.2f}  max={scores.max():.2f}")

    # 各技能样本数
    skill_dist = Counter(s["skill"] for s in samples)
    print("\n  各技能样本数：")
    for skill, n in sorted(skill_dist.items()):
        skill_scores = [s["evaluator_score"] for s in samples if s["skill"] == skill]
        print(f"    {skill:25s}  n={n:3d}  "
              f"avg={np.mean(skill_scores):.2f}  std={np.std(skill_scores):.2f}")

    # 预期 MAE 下界（可以用真实 θ 和 ability_mean_after 算）
    if all("true_theta" in s for s in samples):
        thetas = np.array([s["true_theta"] for s in samples])
        means  = np.array([s["ability_mean_after"] for s in samples])
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(thetas, means)
        print(f"\n  AbilityModel mean vs true_theta MAE：{mae:.4f}")

    print("=" * 60 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# 工具
# ═════════════════════════════════════════════════════════════════════════════

def _save_json(data: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(path)


# ═════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="实验一数据生成：LLM 候选人模拟 + LLM 评分 → annotated_samples"
    )
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API Key（默认读取 ANTHROPIC_API_KEY 环境变量）")
    parser.add_argument("--candidates", type=int, default=20,
                        help="候选人数量（默认 20）")
    parser.add_argument("--skills", nargs="+", default=None,
                        help="要考察的技能列表（默认使用内置 5 个技能）")
    parser.add_argument("--questions-per-skill", type=int, default=3,
                        help="每个技能出几道题（默认 3）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--output", default="exp1_data.json",
                        help="输出文件路径（默认 exp1_data.json）")
    parser.add_argument("--append", action="store_true",
                        help="追加到已有数据文件（断点续传）")
    parser.add_argument("--report-only", action="store_true",
                        help="不生成新数据，仅打印现有数据的质量报告")
    args = parser.parse_args()

    # ── 仅报告模式 ────────────────────────────────────────────────────────
    if args.report_only:
        if not Path(args.output).exists():
            print(f"[ERROR] 文件不存在：{args.output}")
            sys.exit(1)
        with open(args.output, encoding="utf-8") as f:
            samples = json.load(f)
        print_data_report(samples)
        return

    # ── API Key 检查 ──────────────────────────────────────────────────────
    if not args.api_key:
        print("[ERROR] 未提供 API Key。请设置 ANTHROPIC_API_KEY 环境变量或使用 --api-key。")
        sys.exit(1)

    client = ClaudeClient(api_key=args.api_key)

    # ── 断点续传文件 ──────────────────────────────────────────────────────
    progress_file = args.output if args.append else None
    if args.append and not Path(args.output).exists():
        log.info("--append 指定的文件不存在，将从头开始生成")
        progress_file = args.output

    # ── 生成 ──────────────────────────────────────────────────────────────
    log.info(
        "开始生成：candidates=%d  skills=%s  q/skill=%d",
        args.candidates,
        args.skills or DEFAULT_SKILLS,
        args.questions_per_skill,
    )

    samples = generate_exp1_data(
        client=client,
        n_candidates=args.candidates,
        skills=args.skills,
        questions_per_skill=args.questions_per_skill,
        seed=args.seed,
        progress_file=progress_file or args.output,   # 总是开启断点续传
    )

    # ── 保存最终结果 ──────────────────────────────────────────────────────
    _save_json(samples, args.output)
    log.info("✓ 已保存 %d 条样本到 %s", len(samples), args.output)

    # ── 质量报告 ──────────────────────────────────────────────────────────
    print_data_report(samples)

    # ── 提示下一步 ────────────────────────────────────────────────────────
    print(f"下一步：python experiment1_ability_model.py --data {args.output} --save exp1_results.json")


if __name__ == "__main__":
    main()