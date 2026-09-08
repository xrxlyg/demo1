"""
interview_orchestrator.py  (v4)

修复项:
    Fix 1: current_difficulty 传入 PolicyInput，policy reasoning 与实际难度一致
    Fix 2: is_followup=True 时跳过 coverage guard，保护追问逻辑
    Fix 3: experience coverage 加入 fuzzy 关键词匹配，解决术语不完全匹配问题
    Fix 4: coverage 统计改为 observations > 0，出题但未评分不算覆盖
    Fix 5: adaptive difficulty 已在 Fix 1 中统一，orchestrator 不再单独 replace
"""

import os
import json
import random
import logging
from dataclasses import replace
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from interview_state0 import InterviewState
from interview_policy import InterviewPolicy, PolicyInput
from question_selector import QuestionTarget

from agents.answer_evaluator_agent import AnswerEvaluatorAgent
from agents.technical_question_agent import TechnicalQuestionAgent
from agents.competency_question_agent import CompetencyQuestionAgent

logger = logging.getLogger(__name__)


# ── 常量 ──────────────────────────────────────────────────────────────────────

MAX_TECHNICAL_QUESTIONS   = 12
MAX_COMPETENCY_QUESTIONS  = 8
MAX_TOTAL_QUESTIONS       = 25

EXPERIENCE_COVERAGE_THRESHOLD = 0.70

DIFFICULTY_UP_THRESHOLD   = 8.0
DIFFICULTY_DOWN_THRESHOLD = 5.0
DIFFICULTY_LADDER = ["easy", "medium", "hard"]

OPENING_QUESTION = (
    "请先简单介绍一下您最近的一段工作或项目经历，"
    "以及您在其中承担的主要职责。"
)
CLOSING_STATEMENT = (
    "非常感谢您今天参加面试，我们对您的经历和技能有了很好的了解。"
    "面试到此结束，您有什么想问我的吗？"
)

# Fix 3: 模糊匹配词典 — 候选人可能用的别称 → 标准技能名
# 用于 experience coverage 的技能命中判断
SKILL_ALIASES: Dict[str, List[str]] = {
    "redis":        ["缓存", "cache", "redis"],
    "kafka":        ["消息队列", "message queue", "mq", "kafka", "消息中间件"],
    "docker":       ["容器", "container", "docker", "镜像"],
    "kubernetes":   ["k8s", "kubernetes", "容器编排", "orchestration"],
    "mysql":        ["mysql", "关系型数据库", "rdb", "sql数据库"],
    "postgresql":   ["postgresql", "postgres", "pg"],
    "mongodb":      ["mongodb", "mongo", "文档数据库", "nosql"],
    "elasticsearch":["elasticsearch", "es", "全文检索", "搜索引擎"],
    "spring":       ["spring", "spring框架", "ioc", "aoc"],
    "spring boot":  ["spring boot", "springboot"],
    "分布式系统":    ["分布式", "distributed", "微服务", "microservice"],
    "消息队列":      ["消息队列", "message queue", "mq", "kafka", "rabbitmq", "rocketmq"],
    "数据库":        ["数据库", "database", "db", "mysql", "postgresql", "oracle"],
    "缓存":          ["缓存", "cache", "redis", "memcached"],
    "多线程":        ["多线程", "并发", "concurrent", "thread", "线程池"],
    "系统架构":      ["架构", "architecture", "系统设计", "system design"],
    "性能优化":      ["性能优化", "performance", "优化", "调优", "tuning"],
    "设计模式":      ["设计模式", "design pattern", "pattern", "策略模式", "工厂模式"],
    "git":          ["git", "版本控制", "version control", "svn"],
    "ci/cd":        ["ci/cd", "cicd", "持续集成", "continuous integration", "jenkins", "pipeline"],
    "docker":       ["docker", "容器", "container"],
}


def _skill_fuzzy_match(skill: str, text: str) -> bool:
    """
    Fix 3: 判断 skill 是否在 text 中（支持别称）。
    text 通常是 experience_unit 的 summary / responsibility 字段拼接。

    策略:
        1. 精确子串匹配（最快）
        2. 查 SKILL_ALIASES 表，用别称匹配
    """
    skill_lower = skill.lower()
    text_lower  = text.lower()

    # 1. 精确包含
    if skill_lower in text_lower:
        return True

    # 2. 别称匹配
    aliases = SKILL_ALIASES.get(skill_lower, [])
    return any(alias in text_lower for alias in aliases)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class InterviewOrchestrator:

    def __init__(
        self,
        llm_wrapper,
        skill_pool: List[str],
        dimension_pool: List[str],
        dimension_tree: Dict[str, Any],
        min_skills: int = 1,
        max_skills: int = 3,
        min_dimensions: int = 1,
        max_dimensions: int = 3,
        save_training_data: bool = False,
        training_data_dir: str = "training_data",
        candidate_name: str = "candidate",
        max_questions: int = MAX_TOTAL_QUESTIONS,
        policy: Optional[InterviewPolicy] = None,
        evaluator: Optional[AnswerEvaluatorAgent] = None,
        tech_agent: Optional[TechnicalQuestionAgent] = None,
        comp_agent: Optional[CompetencyQuestionAgent] = None,
    ):
        self.llm_wrapper       = llm_wrapper
        self.skill_pool        = skill_pool
        self.dimension_pool    = dimension_pool
        self.dimension_tree    = dimension_tree

        self.min_skills        = min_skills
        self.max_skills        = max_skills
        self.min_dimensions    = min_dimensions
        self.max_dimensions    = max_dimensions
        self.max_questions     = max_questions

        self.save_training_data = save_training_data
        self.training_data_dir  = training_data_dir
        self.candidate_name     = candidate_name

        self.state      = InterviewState()
        self.policy     = policy     or InterviewPolicy()
        self.evaluator  = evaluator  or AnswerEvaluatorAgent(llm_wrapper)
        self.tech_agent = tech_agent or TechnicalQuestionAgent(llm_wrapper)
        self.comp_agent = comp_agent or CompetencyQuestionAgent(llm_wrapper)

        self._current_question:   str  = ""
        self._finished:           bool = False
        self._current_difficulty: str  = "medium"

        # Fix 3: experience coverage，值为已命中技能 set
        self._experience_skill_hits: Dict[int, set] = {}

        # Fix 4: coverage guard，记录"已出题"的 skill/dimension
        #        但覆盖率统计改为 observations > 0（见 _compute_*_coverage）
        self._skills_asked:     set = set()
        self._dimensions_asked: set = set()

        # 公开数据容器（example_usage.py 直接访问）
        self.required_skills:     List[str]      = []
        self.required_dimensions: List[Dict]     = []
        self.job_name:            str            = ""

        self.answer_evaluations:   List[Dict]    = []
        self.question_evaluations: List[Dict]    = []
        self.training_samples:     List[Dict]    = []
        self.confusion_history:    List[Dict]    = []

        self.skill_coverage:     Dict[str, Any] = {}
        self.dimension_coverage: Dict[str, Any] = {}

    # =========================================================================
    # 公开接口
    # =========================================================================

    def initialize_interview(self, job_profile: Dict[str, Any]) -> None:
        """解析 job_profile，选取技能/维度，初始化 InterviewState。"""
        self.job_name = job_profile.get("job_name", "未知岗位")

        raw_text = job_profile.get("raw_text", "").lower()
        matched  = [s for s in self.skill_pool if s.lower() in raw_text]
        if len(matched) < self.min_skills:
            extras  = [s for s in self.skill_pool if s not in matched]
            matched += random.sample(extras, min(self.min_skills - len(matched), len(extras)))
        self.required_skills = matched[: self.max_skills]

        n_dims = random.randint(self.min_dimensions, self.max_dimensions)
        chosen = random.sample(self.dimension_pool, min(n_dims, len(self.dimension_pool)))
        self.required_dimensions = [
            {"dimension": d, "sub_dimensions": self.dimension_tree.get(d, [])}
            for d in chosen
        ]

        job_analysis = {
            "job_name":              self.job_name,
            "job_id":                job_profile.get("job_id", ""),
            "recruit_type":          job_profile.get("recruit_type", ""),
            "required_skills":       self.required_skills,
            "preferred_skills":      [],
            "competency_dimensions": self.required_dimensions,
            "raw_text":              job_profile.get("raw_text", ""),
        }
        self.state.initialize(job_analysis)
        self.state.interview_phase = "opening"

        logger.info("Interview initialized | job=%s skills=%s dimensions=%s",
                    self.job_name, self.required_skills,
                    [d["dimension"] for d in self.required_dimensions])

    def generate_opening_question(self) -> str:
        self._current_question = OPENING_QUESTION
        return OPENING_QUESTION

    def process_response(
        self,
        answer: str,
        question: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict]]:
        """
        处理候选人回答，返回 (下一题, 本轮评估)。

        Args:
            answer:   候选人回答
            question: 本轮问题（可选，默认用内部 _current_question）
        """
        if question:
            self._current_question = question

        if self._finished:
            return CLOSING_STATEMENT, None

        evaluation: Optional[Dict] = None
        current_phase = self.state.interview_phase

        # ── Step 1: opening ───────────────────────────────────────────────
        if current_phase == "opening":
            self._handle_opening_response(answer)
            self.state.interview_phase = "technical"

        # ── Step 2: 评估 + 记录 ───────────────────────────────────────────
        else:
            evaluation = self._evaluate_and_record(self._current_question, answer)
            self._adapt_difficulty(self.state.last_answer_score)

        # ── Step 3: 硬终止 ────────────────────────────────────────────────
        if self._should_end_interview():
            self._finished = True
            self._maybe_save_training_data()
            self._refresh_coverage_views()
            return CLOSING_STATEMENT, evaluation

        # ── Step 4: Policy 选题（Fix 1: 传入 current_difficulty）─────────
        target = self._run_policy()

        # ── Step 5: Coverage guard（Fix 2: followup 时跳过）──────────────
        if not target.is_followup:
            target = self._apply_coverage_guard(target)

        # ── Step 6: Phase 同步 ────────────────────────────────────────────
        self._sync_phase(target)

        if self.state.interview_phase == "closing":
            self._finished = True
            self._maybe_save_training_data()
            self._refresh_coverage_views()
            return CLOSING_STATEMENT, evaluation

        # ── Step 7: Experience coverage（Fix 3: fuzzy match）─────────────
        self._maybe_advance_experience(target)

        # ── Step 8: 生成问题 ──────────────────────────────────────────────
        next_question = self._generate_question(target)

        # ── Step 9: 记录已出题 ────────────────────────────────────────────
        self._record_question_asked(target)

        # ── Step 10: 更新追踪字段 ─────────────────────────────────────────
        self._current_question         = next_question
        self.state.last_question_skill = target.target
        self.state.last_question_type  = target.question_type

        if target.is_followup:
            self.state.followup_counts[target.target] = (
                self.state.followup_counts.get(target.target, 0) + 1
            )

        self._refresh_coverage_views()

        logger.info("Q#%d | phase=%s skill=%s followup=%s difficulty=%s",
                    self.state.question_count + 1,
                    self.state.interview_phase,
                    target.target, target.is_followup, target.difficulty)

        return next_question, evaluation

    def generate_overall_evaluation(self) -> Dict[str, Any]:
        ability = self.state.ability_model
        skill_scores = {s: round(a.mean * 10, 1)
                        for s, a in ability.skills.items() if a.observations > 0}
        dim_scores   = {d: round(a.mean * 10, 1)
                        for d, a in ability.dimensions.items() if a.observations > 0}
        all_obs  = list(ability.skills.values()) + list(ability.dimensions.values())
        assessed = [a.mean for a in all_obs if a.observations > 0]
        mean     = sum(assessed) / len(assessed) if assessed else 0.5
        return {
            "candidate_name":   self.candidate_name,
            "job_name":         self.job_name,
            "interview_date":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_level":    self._mean_to_level(mean),
            "overall_score":    round(mean * 10, 1),
            "skill_scores":     skill_scores,
            "dimension_scores": dim_scores,
            "phases_completed": self._infer_phases_completed(),
            "ability_state":    ability.summary(),
            "recommendation":   self._make_recommendation(mean),
        }

    def get_interview_summary(self) -> Dict[str, Any]:
        cov_s = self._compute_skill_coverage()
        cov_d = self._compute_dimension_coverage()
        skill_detail = {
            s: [i for i, m in enumerate(self.state.question_metadata)
                if m.get("skill") == s]
            for s in self.required_skills
        }
        dim_detail = {
            d["dimension"]: [
                i for i, m in enumerate(self.state.question_metadata)
                if m.get("skill") == d["dimension"]
            ]
            for d in self.required_dimensions
        }
        return {
            "总问题数":      self.state.question_count,
            "讨论经历数":    self.state.current_experience_index + 1,
            "覆盖技能数":    cov_s["covered"],
            "评估维度数":    cov_d["covered"],
            "当前难度":      self._current_difficulty,
            "最终阶段":      self.state.interview_phase,
            "技能覆盖率":    f"{cov_s['rate'] * 100:.0f}%",
            "维度覆盖率":    f"{cov_d['rate'] * 100:.0f}%",
            "详细覆盖情况":  {"技能": skill_detail, "维度": dim_detail},
        }

    # =========================================================================
    # Fix 1: _run_policy — 传入 current_difficulty
    # =========================================================================

    def _run_policy(self) -> QuestionTarget:
        """
        构造 PolicyInput，传入 current_difficulty。
        Policy 内部直接使用此值调用 selector，reasoning 与实际难度完全一致。
        不再需要事后 replace(target, difficulty=...)。
        """
        policy_input = PolicyInput(
            ability_model=self.state.ability_model,
            skill_graph=self.state.skill_graph,
            dialogue_history=self.state.dialogue_history,
            interview_phase=self.state.interview_phase,
            question_count=self.state.question_count,
            last_answer_score=self.state.last_answer_score,
            last_question_skill=self.state.last_question_skill,
            followup_counts=self.state.followup_counts,
            current_difficulty=self._current_difficulty,   # Fix 1
        )
        return self.policy.select_next_question(policy_input)

    # =========================================================================
    # Fix 2: Coverage guard 跳过 followup（已在 process_response Step 5 实现）
    # =========================================================================

    def _apply_coverage_guard(self, target: QuestionTarget) -> QuestionTarget:
        """
        Fix 2: 此方法只在 target.is_followup == False 时被调用（见 Step 5）。
        若 policy 想追问（is_followup=True），orchestrator 直接跳过此方法，
        保护追问逻辑不被覆盖率重定向破坏。
        """
        if target.question_type == "technical":
            asked     = self._skills_asked
            uncovered = [
                s for s, a in self.state.ability_model.skills.items()
                if a.observations == 0 and s not in asked
            ]
        else:
            asked     = self._dimensions_asked
            uncovered = [
                d for d, a in self.state.ability_model.dimensions.items()
                if a.observations == 0 and d not in asked
            ]

        if target.target in asked and uncovered:
            forced = uncovered[0]
            logger.info("Coverage guard: %s → %s", target.target, forced)
            return replace(
                target,
                target=forced,
                is_followup=False,
                reasoning=f"coverage guard: forced '{forced}'",
            )
        return target

    # =========================================================================
    # Fix 3: Experience coverage — fuzzy skill matching
    # =========================================================================

    def _maybe_advance_experience(self, target: QuestionTarget) -> None:
        idx        = self.state.current_experience_index
        experience = self.state.get_current_experience()
        if not experience:
            return

        exp_skills = self._get_experience_skills(experience)
        if not exp_skills:
            if self._experience_skill_hits.get(idx):
                self.state.move_to_next_experience()
            return

        hit_set  = self._experience_skill_hits.get(idx, set())
        coverage = len(hit_set) / len(exp_skills)
        if coverage >= EXPERIENCE_COVERAGE_THRESHOLD:
            self.state.move_to_next_experience()
            logger.info("Advanced experience %d (coverage %.0f%%)", idx + 1, coverage * 100)

    def _get_experience_skills(self, experience: Dict) -> List[str]:
        """提取经历单元中的所有技能标签（tech_stack + methods）。"""
        return list(set(
            experience.get("tech_stack", []) + experience.get("methods", [])
        ))

    def _update_experience_hit(self, target: QuestionTarget) -> None:
        """
        Fix 3: 判断本题 target.target 是否命中当前经历中的某个技能。
        使用 fuzzy matching（精确匹配 + SKILL_ALIASES 别称表）。
        """
        idx        = self.state.current_experience_index
        experience = self.state.get_current_experience()
        if not experience:
            return

        exp_skills = self._get_experience_skills(experience)

        # 构造用于模糊匹配的上下文文本（技能标签 + summary + responsibility）
        exp_text = " ".join(
            exp_skills
            + [experience.get("summary", "")]
            + [experience.get("claimed_responsibility", "")]
        )

        # 精确命中：target 在 exp_skills 列表中
        if target.target in exp_skills:
            self._experience_skill_hits.setdefault(idx, set()).add(target.target)
            return

        # 模糊命中：target 的别称出现在经历文本中，视为命中 target
        if _skill_fuzzy_match(target.target, exp_text):
            self._experience_skill_hits.setdefault(idx, set()).add(target.target)
            logger.debug("Fuzzy hit: '%s' in experience %d", target.target, idx)

    def _record_question_asked(self, target: QuestionTarget) -> None:
        if target.question_type == "technical":
            self._skills_asked.add(target.target)
        else:
            self._dimensions_asked.add(target.target)
        self._update_experience_hit(target)

    # =========================================================================
    # Fix 4: Coverage 统计改为 observations > 0
    # =========================================================================

    def _compute_skill_coverage(self) -> Dict[str, Any]:
        """
        Fix 4: covered = observations > 0（候选人真正回答过，而非仅出题）。
        _skills_asked 仅用于 coverage guard 去重，不再作为覆盖率分子。
        """
        skills = self.state.ability_model.skills
        total  = len(skills)
        covered_names = [s for s, a in skills.items() if a.observations > 0]
        covered = len(covered_names)
        return {
            "total":     total,
            "covered":   covered,
            "rate":      round(covered / total, 2) if total else 0.0,
            "uncovered": [s for s in skills if s not in covered_names],
        }

    def _compute_dimension_coverage(self) -> Dict[str, Any]:
        """Fix 4: 同上，observations > 0 才算真正覆盖。"""
        dims   = self.state.ability_model.dimensions
        total  = len(dims)
        covered_names = [d for d, a in dims.items() if a.observations > 0]
        covered = len(covered_names)
        return {
            "total":     total,
            "covered":   covered,
            "rate":      round(covered / total, 2) if total else 0.0,
            "uncovered": [d for d in dims if d not in covered_names],
        }

    def _refresh_coverage_views(self) -> None:
        self.skill_coverage     = self._compute_skill_coverage()
        self.dimension_coverage = self._compute_dimension_coverage()

    # =========================================================================
    # Adaptive Difficulty
    # =========================================================================

    def _adapt_difficulty(self, score: Optional[float]) -> None:
        """
        根据上一题得分动态调整 self._current_difficulty。
        下一轮 _run_policy 会将新难度通过 PolicyInput.current_difficulty 传入，
        保证 policy reasoning 与实际难度完全一致（Fix 1 的配套）。
        """
        if score is None:
            return
        idx = DIFFICULTY_LADDER.index(self._current_difficulty)
        if score > DIFFICULTY_UP_THRESHOLD:
            new_idx = min(idx + 1, len(DIFFICULTY_LADDER) - 1)
        elif score < DIFFICULTY_DOWN_THRESHOLD:
            new_idx = max(idx - 1, 0)
        else:
            return
        if new_idx != idx:
            old = self._current_difficulty
            self._current_difficulty = DIFFICULTY_LADDER[new_idx]
            logger.info("Difficulty: %s → %s (score=%.1f)", old, self._current_difficulty, score)

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _handle_opening_response(self, answer: str) -> None:
        self.state.add_dialogue(
            question=self._current_question,
            answer=answer,
            skill=None,
            question_type="opening",
            is_followup=False,
            difficulty="easy",
        )

    def _evaluate_and_record(self, question: str, answer: str) -> Dict:
        question_type = self.state.last_question_type or "technical"
        target_skill  = self.state.last_question_skill or ""

        try:
            evaluation = self.evaluator.run(
                question=question,
                answer=answer,
                question_type=question_type,
                target_skill_or_dimension=target_skill,
            )
        except Exception as e:
            logger.warning("Evaluator failed: %s. score=5.0", e)
            evaluation = {
                "评分":     5.0,
                "评估类型": "技术评估" if question_type == "technical" else "能力评估",
                "考察目标": target_skill,
                "评语":     "评估失败，使用默认分数",
            }

        self.state.record_evaluation(evaluation)

        is_followup = (
            self.state.followup_counts.get(target_skill, 0) > 0
            and self.state.last_question_skill == target_skill
        )
        self.state.add_dialogue(
            question=question,
            answer=answer,
            skill=target_skill,
            question_type=question_type,
            is_followup=is_followup,
            difficulty=self._current_difficulty,
        )

        self.answer_evaluations.append(evaluation)

        if self.save_training_data:
            self.training_samples.append({
                "candidate":     self.candidate_name,
                "job_name":      self.job_name,
                "question":      question,
                "answer":        answer,
                "question_type": question_type,
                "skill":         target_skill,
                "difficulty":    self._current_difficulty,
                "score":         evaluation.get("评分"),
                "evaluation":    evaluation,
            })

        return evaluation

    def _generate_question(self, target: QuestionTarget) -> str:
        experience = self.state.get_current_experience() or {}
        if target.question_type == "technical":
            return self._generate_technical(target, experience)
        return self._generate_competency(target, experience)

    def _generate_technical(self, target: QuestionTarget, experience: Dict) -> str:
        covered = [s for s, a in self.state.ability_model.skills.items()
                   if a.observations >= 2]
        try:
            return self.tech_agent.run(
                experience_unit=experience,
                covered_tech_points=covered,
                difficulty_level=target.difficulty,
            )
        except Exception as e:
            logger.error("TechnicalQuestionAgent failed: %s", e)
            return f"请介绍一下您在 {target.target} 方面的理解和实践经验。"

    def _generate_competency(self, target: QuestionTarget, experience: Dict) -> str:
        dimension_dict = self._find_dimension_dict(target.target)
        try:
            return self.comp_agent.run(
                experience_unit=experience,
                target_dimension=dimension_dict,
                dialogue_history=self.state.dialogue_history,
                difficulty_level=target.difficulty,
            )
        except Exception as e:
            logger.error("CompetencyQuestionAgent failed: %s", e)
            return f"请描述一个体现您{target.target}能力的具体经历。"

    def _sync_phase(self, target: QuestionTarget) -> None:
        current   = self.state.interview_phase
        next_type = target.question_type
        if next_type == "closing":
            self.state.interview_phase = "closing"
        elif next_type == "competency" and current == "technical":
            self.state.interview_phase = "competency"
            logger.info("Phase transition: technical → competency")
        elif next_type == "technical" and current == "opening":
            self.state.interview_phase = "technical"

    def _should_end_interview(self) -> bool:
        q = self.state.question_count
        if q >= self.max_questions:
            logger.info("Hard limit: %d/%d", q, self.max_questions)
            return True
        phase  = self.state.interview_phase
        tech_q = sum(1 for m in self.state.question_metadata if m.get("question_type") == "technical")
        comp_q = sum(1 for m in self.state.question_metadata if m.get("question_type") == "competency")
        if phase == "technical"  and tech_q >= MAX_TECHNICAL_QUESTIONS:
            return False
        if phase == "competency" and comp_q >= MAX_COMPETENCY_QUESTIONS:
            return True
        return False

    def _maybe_save_training_data(self) -> None:
        if not self.save_training_data or not self.training_samples:
            return
        try:
            os.makedirs(self.training_data_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d")
            filename = os.path.join(
                self.training_data_dir,
                f"{date_str}_{self.candidate_name}_{self.job_name}.json",
            )
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.training_samples, f, ensure_ascii=False, indent=2)
            logger.info("Training data saved: %s (%d records)", filename, len(self.training_samples))
        except Exception as e:
            logger.error("Failed to save training data: %s", e)

    def _find_dimension_dict(self, dimension_name: str) -> Dict:
        for dim in self.state.job_analysis.get("competency_dimensions", []):
            if isinstance(dim, dict) and dim.get("dimension") == dimension_name:
                return dim
            if isinstance(dim, str) and dim == dimension_name:
                return {"dimension": dimension_name, "sub_dimensions": []}
        return {"dimension": dimension_name, "sub_dimensions": []}

    def _mean_to_level(self, mean: float) -> str:
        if mean >= 0.78:   return "senior"
        elif mean >= 0.52: return "mid"
        return "junior"

    def _make_recommendation(self, mean: float) -> str:
        if mean >= 0.78:   return "强烈推荐录用"
        elif mean >= 0.60: return "推荐录用"
        elif mean >= 0.45: return "有条件录用，建议进一步评估"
        return "暂不推荐"

    def _infer_phases_completed(self) -> List[str]:
        types_seen = {m.get("question_type") for m in self.state.question_metadata}
        phases = ["opening"]
        if "technical"  in types_seen: phases.append("technical")
        if "competency" in types_seen: phases.append("competency")
        if self._finished:             phases.append("closing")
        return phases

    # =========================================================================
    # 属性
    # =========================================================================

    @property
    def current_question(self) -> str:
        return self._current_question

    @property
    def progress(self) -> Dict[str, Any]:
        summary = self.state.get_progress_summary()
        summary.update({
            "candidate_name":     self.candidate_name,
            "current_difficulty": self._current_difficulty,
            "skill_coverage":     self._compute_skill_coverage(),
            "dimension_coverage": self._compute_dimension_coverage(),
            "max_questions":      self.max_questions,
        })
        return summary
