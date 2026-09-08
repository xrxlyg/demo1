"""
question_selector.py  (v6)

v5 → v6 核心变更：
    ✅ 删除内部 AbilityEstimate / AbilityModel 定义
       改为直接使用 ability_model.py 中的 AbilityModel / SkillAbility
    ✅ 所有读取不确定性的地方从 .std 改为 .uncertainty
    ✅ information_gain() 直接调用 SkillAbility.information_gain()
    ✅ 其余评分逻辑（SP / branch / UCB）完整保留
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── 使用项目统一的 AbilityModel ──────────────────────────────────────────────
from ability_model import AbilityModel, SkillAbility

logger = logging.getLogger(__name__)


# ── Difficulty target ─────────────────────────────────────────────────────────
DIFFICULTY_TARGET: Dict[str, float] = {"easy": 0.30, "medium": 0.55, "hard": 0.80}
DEFAULT_WEIGHTS = dict(alpha=0.35, beta=0.30, gamma=0.20, delta=0.15)

# ── Branch hyper-parameters ───────────────────────────────────────────────────
LAMBDA_BRANCH_PENALTY: float = 0.5
ETA_BRANCH_BONUS:      float = 0.10
KAPPA_UCB:             float = 0.10

# ── Structural Prior hyper-parameters ────────────────────────────────────────
SP_WEIGHT_DEPTH:   float = 0.12
SP_WEIGHT_BREADTH: float = 0.08
SP_WEIGHT_ORDER:   float = 0.05
SP_DECAY_STEPS:    int   = 5


# ── QuestionTarget ────────────────────────────────────────────────────────────

@dataclass
class QuestionTarget:
    target:        str
    question_type: str   = "technical"
    difficulty:    str   = "medium"
    is_followup:   bool  = False
    score:         float = 0.0
    reasoning:     str   = ""
    components:    Dict  = field(default_factory=dict)


# ── v3 Scoring Functions ──────────────────────────────────────────────────────

def information_gain(e: SkillAbility) -> float:
    """Delegate to SkillAbility's own method, cap at 1."""
    return min(e.information_gain(), 1.0)

def coverage_gain(e: SkillAbility) -> float:
    """CG = 1 if never asked, else decays with observations."""
    return 1.0 if e.observations == 0 else max(0.0, 1.0 - 0.2 * e.observations)

def difficulty_match(e: SkillAbility, target_difficulty: str) -> float:
    """DM = max(0, 1 - 2|μ - t_d|)"""
    t_d = DIFFICULTY_TARGET.get(target_difficulty, 0.55)
    return max(0.0, 1.0 - 2.0 * abs(e.mean - t_d))

def dialogue_relevance(skill: str, recent: List[str]) -> float:
    if not recent:           return 1.0
    if skill == recent[-1]:  return 0.0
    if skill in recent[-4:]: return 0.4
    return 1.0

def score_question(
    skill: str, estimate: SkillAbility,
    recent_skills: List[str], target_difficulty: str,
    alpha=DEFAULT_WEIGHTS["alpha"], beta=DEFAULT_WEIGHTS["beta"],
    gamma=DEFAULT_WEIGHTS["gamma"], delta=DEFAULT_WEIGHTS["delta"],
) -> Tuple[float, Dict]:
    ig = information_gain(estimate)
    cg = coverage_gain(estimate)
    dm = difficulty_match(estimate, target_difficulty)
    dr = dialogue_relevance(skill, recent_skills)
    total = alpha*ig + beta*cg + gamma*dm + delta*dr
    return total, {
        "IG": round(ig, 4), "CG": round(cg, 4),
        "DM": round(dm, 4), "DR": round(dr, 4),
        "total": round(total, 4),
    }


# ── Structural Prior ──────────────────────────────────────────────────────────

def _lookup_node(name: str, skill_graph):
    if hasattr(skill_graph, "nodes"): return skill_graph.nodes.get(name)
    if hasattr(skill_graph, "get"):   return skill_graph.get(name)
    return None

def _node_metadata(skill: str, skill_graph) -> Tuple[int, int, int]:
    if skill_graph is None: return 1, 1, 0
    node = _lookup_node(skill, skill_graph)
    if node is None: return 1, 1, 0
    depth = getattr(node, "depth", 1)
    subtree_size = 1
    if hasattr(skill_graph, "get_descendants"):
        subtree_size = 1 + len(skill_graph.get_descendants(skill))
    sibling_index = 0
    parent = getattr(node, "parent", None)
    if parent and hasattr(skill_graph, "children_of"):
        names = [s.name for s in skill_graph.children_of(parent)]
        sibling_index = names.index(skill) if skill in names else 0
    return depth, subtree_size, sibling_index

def structural_prior(
    skill: str, estimate: SkillAbility, skill_graph,
    max_depth: int = 4, max_breadth: int = 10, max_siblings: int = 8,
) -> float:
    decay = max(0.0, 1.0 - estimate.observations / SP_DECAY_STEPS)
    if decay == 0.0 or skill_graph is None:
        return 0.0
    depth, subtree_size, sibling_index = _node_metadata(skill, skill_graph)
    depth_score   = 1.0 - min(depth, max_depth)    / max_depth
    breadth_score = min(subtree_size, max_breadth)  / max_breadth
    order_score   = 1.0 - min(sibling_index, max_siblings) / max_siblings
    return decay * (
        SP_WEIGHT_DEPTH   * depth_score
        + SP_WEIGHT_BREADTH * breadth_score
        + SP_WEIGHT_ORDER   * order_score
    )


# ── Branch Utilities ──────────────────────────────────────────────────────────

def _get_branch(skill: str, skill_graph) -> Optional[str]:
    if skill_graph is None: return None
    node = _lookup_node(skill, skill_graph)
    if node is None: return None
    if getattr(node, "depth", 1) <= 1: return node.name
    current = node
    while current.parent:
        p = _lookup_node(current.parent, skill_graph)
        if p is None: break
        if getattr(p, "depth", 0) == 1: return p.name
        current = p
    return current.name

def _branch_freq(skill: str, recent: List[str], skill_graph, window: int = 6) -> float:
    if not recent or skill_graph is None: return 0.0
    tb = _get_branch(skill, skill_graph)
    if tb is None: return 0.0
    r = recent[-window:]
    return sum(1 for s in r if _get_branch(s, skill_graph) == tb) / len(r)

def _branch_coverage(branch: str, skill_graph, am: AbilityModel) -> float:
    if skill_graph is None: return 1.0
    desc = skill_graph.get_descendants(branch) if hasattr(skill_graph, "get_descendants") else []
    if not desc:
        return 1.0 if am.skills.get(branch, SkillAbility()).observations > 0 else 0.0
    covered = sum(1 for s in desc if am.skills.get(s, SkillAbility()).observations > 0)
    return covered / len(desc)

def branch_penalty_factor(skill: str, recent: List[str], skill_graph) -> float:
    return math.exp(-LAMBDA_BRANCH_PENALTY * _branch_freq(skill, recent, skill_graph))

def branch_coverage_bonus(skill: str, skill_graph, am: AbilityModel) -> float:
    if skill_graph is None: return 0.0
    tb = _get_branch(skill, skill_graph)
    if tb is None: return 0.0
    return ETA_BRANCH_BONUS * (1.0 - _branch_coverage(tb, skill_graph, am))

def ucb_bonus(skill: str, am: AbilityModel, T: int) -> float:
    N = am.skills.get(skill, SkillAbility()).observations
    return KAPPA_UCB * math.sqrt(math.log(max(T, 1)) / (1.0 + N))


# ── Combined Scoring ──────────────────────────────────────────────────────────

def score_question_full(
    skill: str, estimate: SkillAbility, recent_skills: List[str],
    target_difficulty: str, ability_model: AbilityModel,
    total_questions: int, skill_graph=None,
    alpha=DEFAULT_WEIGHTS["alpha"], beta=DEFAULT_WEIGHTS["beta"],
    gamma=DEFAULT_WEIGHTS["gamma"], delta=DEFAULT_WEIGHTS["delta"],
) -> Tuple[float, Dict]:
    v3_base, comps = score_question(skill, estimate, recent_skills,
                                    target_difficulty, alpha, beta, gamma, delta)
    sp      = structural_prior(skill, estimate, skill_graph)
    bb      = branch_coverage_bonus(skill, skill_graph, ability_model)
    ucb     = ucb_bonus(skill, ability_model, total_questions)
    penalty = branch_penalty_factor(skill, recent_skills, skill_graph)
    final   = (v3_base + sp + bb + ucb) * penalty
    comps.update({
        "v3_base":        round(v3_base, 4),
        "SP":             round(sp, 4),
        "branch_bonus":   round(bb, 4),
        "ucb_bonus":      round(ucb, 4),
        "branch_penalty": round(penalty, 4),
        "total":          round(final, 4),
    })
    return final, comps


# ── QuestionSelector ──────────────────────────────────────────────────────────

class QuestionSelector:

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 followup_boost: float = 0.15) -> None:
        self.weights        = weights or DEFAULT_WEIGHTS.copy()
        self.followup_boost = followup_boost

    def select(
        self, ability_model: AbilityModel, candidates: List[str],
        question_type: str, target_difficulty: str, recent_skills: List[str],
        last_skill: Optional[str] = None,
        followup_counts: Optional[Dict[str, int]] = None,
        max_followup: int = 2,
    ) -> QuestionTarget:
        """Competency / flat argmax — no branch/SP."""
        if not candidates:
            return QuestionTarget(target="unknown", question_type="closing",
                                  reasoning="No candidates available.")
        fc = followup_counts or {}
        w  = self.weights
        scored = []
        for skill in candidates:
            est = (ability_model.skills.get(skill, SkillAbility())
                   if question_type == "technical"
                   else ability_model.dimensions.get(skill, SkillAbility()))
            total, comps = score_question(skill, est, recent_skills, target_difficulty,
                                          w["alpha"], w["beta"], w["gamma"], w["delta"])
            if (skill == last_skill and est.uncertainty > 0.15
                    and fc.get(skill, 0) < max_followup):
                total += self.followup_boost
                comps["followup_boost"] = self.followup_boost
            scored.append((total, skill, comps))
        scored.sort(key=lambda x: x[0], reverse=True)
        bs, bsk, bc = scored[0]
        reasoning = (f"argmax '{bsk}' score={bs:.4f} | "
                     f"IG={bc['IG']:.3f} CG={bc['CG']:.3f} "
                     f"DM={bc['DM']:.3f} DR={bc['DR']:.3f}")
        return QuestionTarget(target=bsk, question_type=question_type,
                              difficulty=target_difficulty, is_followup=(bsk == last_skill),
                              score=bs, reasoning=reasoning, components=bc)

    def select_with_graph(
        self, ability_model: AbilityModel, skill_graph,
        question_type: str, target_difficulty: str, recent_skills: List[str],
        current_skill: Optional[str], traversal_state,
        followup_counts: Optional[Dict[str, int]] = None,
        max_followup: int = 2,
    ) -> QuestionTarget:
        traversal_state.last_score = (
            ability_model.skills.get(current_skill, SkillAbility()).mean
            if current_skill else 0.5)

        if current_skill:
            next_from_graph, strategy = skill_graph.next_node(current_skill, traversal_state)
        else:
            next_from_graph = skill_graph.topic_switch()
            strategy        = "topic_switch_init"

        local: List[str] = []
        if next_from_graph:
            node = skill_graph.get(next_from_graph)
            if node and node.parent:
                local = [n.name for n in skill_graph.children_of(node.parent)]
            else:
                local = [next_from_graph]
        candidates = list(set(local + [r.name for r in skill_graph.roots()]))
        if not candidates:
            candidates = skill_graph.all_names()

        fc = followup_counts or {}
        w  = self.weights
        T  = sum(a.observations for a in ability_model.skills.values()) or 1

        scored = []
        for skill in candidates:
            est   = ability_model.skills.get(skill, SkillAbility())
            total, comps = score_question_full(
                skill, est, recent_skills, target_difficulty,
                ability_model, T, skill_graph,
                w["alpha"], w["beta"], w["gamma"], w["delta"],
            )
            if (skill == current_skill and est.uncertainty > 0.15
                    and fc.get(skill, 0) < max_followup):
                total += self.followup_boost
                comps["followup_boost"] = self.followup_boost
            scored.append((total, skill, comps))

        scored.sort(key=lambda x: x[0], reverse=True)
        bs, bsk, bc = scored[0]
        reasoning = (
            f"[{strategy}] argmax '{bsk}' score={bs:.4f} | "
            f"v3_base={bc.get('v3_base',0):.3f} SP={bc.get('SP',0):.3f} "
            f"branch_b={bc.get('branch_bonus',0):.3f} ucb={bc.get('ucb_bonus',0):.3f} "
            f"penalty={bc.get('branch_penalty',1):.3f}"
        )
        logger.info("select_with_graph | best=%s branch=%s score=%.4f | %s",
                    bsk, _get_branch(bsk, skill_graph), bs, reasoning)
        return QuestionTarget(target=bsk, question_type=question_type,
                              difficulty=target_difficulty, is_followup=(bsk == current_skill),
                              score=bs, reasoning=reasoning, components=bc)


# ── Diagnostic ────────────────────────────────────────────────────────────────

def score_all(
    ability_model: AbilityModel, candidates: List[str],
    question_type: str, target_difficulty: str, recent_skills: List[str],
    weights: Optional[Dict[str, float]] = None, skill_graph=None,
) -> List[Dict]:
    w = weights or DEFAULT_WEIGHTS
    T = sum(a.observations for a in ability_model.skills.values()) or 1
    results = []
    for skill in candidates:
        est = (ability_model.skills.get(skill, SkillAbility())
               if question_type == "technical"
               else ability_model.dimensions.get(skill, SkillAbility()))
        if skill_graph is not None and question_type == "technical":
            total, comps = score_question_full(
                skill, est, recent_skills, target_difficulty,
                ability_model, T, skill_graph, **w)
        else:
            total, comps = score_question(
                skill, est, recent_skills, target_difficulty, **w)
        results.append({"skill": skill, **comps, "estimate": {
            "mean": round(est.mean, 4),
            "uncertainty": round(est.uncertainty, 4),
            "observations": est.observations,
        }})
    results.sort(key=lambda x: x["total"], reverse=True)
    return results