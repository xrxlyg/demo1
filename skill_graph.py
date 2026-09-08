"""
skill_graph.py

Formalized SkillGraph implementation aligned with the method section.

Models the skill space as a directed graph:
    G = (V, E)

Each node:
    v_i = (name_i, parent_i, depth_i, w_i)

Traversal policies:
    - Deep-Dive:    v_{t+1} ∈ children(v_t)
    - Topic-Switch: v_{t+1} = argmin_{r_i ∈ R} n_i

Replaces the flat `required_skills: List[str]` with a proper tree structure
that supports hierarchical dependency modeling and state-driven traversal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ── Node ─────────────────────────────────────────────────────────────────────

@dataclass
class SkillNode:
    """
    v_i = (name_i, parent_i, depth_i, w_i)

    Attributes:
        name:        Skill identifier (unique within the graph).
        parent:      Parent node name, or None for root nodes.
        depth:       Depth in the hierarchy tree (root = 0).
        weight:      Importance weight w_i ∈ (0, 1].
        children:    Names of child nodes (populated by SkillGraph.add_edge).
        observations: n_s — number of times this skill has been assessed.
                      Used by IG / CG calculations and traversal policy.
    """
    name:         str
    parent:       Optional[str]  = None
    depth:        int            = 0
    weight:       float          = 1.0
    children:     List[str]      = field(default_factory=list)
    observations: int            = 0

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ── Graph ─────────────────────────────────────────────────────────────────────

class SkillGraph:
    """
    Directed graph G = (V, E) representing the hierarchical skill topology.

    Hierarchical constraint:
        depth(child) = depth(parent) + 1

    Example structure (Java backend):
        分布式系统 (depth=0, root)
        ├── 一致性协议 (depth=1)
        │   ├── Raft (depth=2)
        │   └── Paxos (depth=2)
        └── 服务发现 (depth=1)
            └── Nacos (depth=2)

    Usage:
        g = SkillGraph()
        g.add_node("分布式系统", weight=1.0)
        g.add_node("一致性协议", weight=0.9)
        g.add_edge("分布式系统", "一致性协议")
        g.add_node("Raft", weight=0.8)
        g.add_edge("一致性协议", "Raft")

    Or use the class method:
        g = SkillGraph.from_dict({
            "分布式系统": {
                "weight": 1.0,
                "children": {
                    "一致性协议": {
                        "weight": 0.9,
                        "children": {
                            "Raft":  {"weight": 0.8},
                            "Paxos": {"weight": 0.7},
                        }
                    }
                }
            }
        })
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, SkillNode] = {}

    # ── Construction ──────────────────────────────────────────────────────────

    def add_node(
        self,
        name:   str,
        weight: float = 1.0,
        parent: Optional[str] = None,
    ) -> SkillNode:
        """Add a skill node. If parent is given, depth is auto-computed."""
        if name in self._nodes:
            return self._nodes[name]

        depth = 0
        if parent is not None:
            if parent not in self._nodes:
                raise ValueError(f"Parent node '{parent}' not found. Add parent first.")
            depth = self._nodes[parent].depth + 1          # depth(child) = depth(parent)+1

        node = SkillNode(name=name, parent=parent, depth=depth, weight=weight)
        self._nodes[name] = node
        return node

    def add_edge(self, parent_name: str, child_name: str) -> None:
        """
        Add directed edge parent → child and enforce the depth constraint.

        If child_name does not yet exist, it is created automatically.
        """
        if parent_name not in self._nodes:
            raise ValueError(f"Parent '{parent_name}' not in graph.")

        parent_node = self._nodes[parent_name]

        if child_name not in self._nodes:
            child_node = SkillNode(
                name=child_name,
                parent=parent_name,
                depth=parent_node.depth + 1,
            )
            self._nodes[child_name] = child_node
        else:
            child_node = self._nodes[child_name]
            # Re-enforce depth constraint if parent changed
            child_node.parent = parent_name
            child_node.depth  = parent_node.depth + 1

        if child_name not in parent_node.children:
            parent_node.children.append(child_name)

    @classmethod
    def from_dict(cls, tree: Dict) -> "SkillGraph":
        """
        Build a SkillGraph from a nested dict.

        Format:
            {
                "skill_name": {
                    "weight": float,          # optional, default 1.0
                    "children": { ... }       # optional
                },
                ...
            }
        """
        g = cls()

        def _recurse(subtree: Dict, parent: Optional[str]) -> None:
            for name, config in subtree.items():
                if not isinstance(config, dict):
                    config = {}
                weight = float(config.get("weight", 1.0))
                g.add_node(name, weight=weight, parent=parent)
                if parent is not None:
                    g.add_edge(parent, name)
                children_dict = config.get("children", {})
                if children_dict:
                    _recurse(children_dict, name)

        _recurse(tree, None)
        return g

    @classmethod
    def from_flat_list(cls, skills: List[str]) -> "SkillGraph":
        """
        Upgrade a flat skill list to a single-level SkillGraph.
        All skills become root nodes (depth=0).
        This preserves backward compatibility with existing job profiles
        while enabling the graph API.
        """
        g = cls()
        for skill in skills:
            g.add_node(skill, weight=1.0, parent=None)
        logger.warning(
            "SkillGraph built from flat list (%d skills). "
            "No hierarchical structure available — consider providing a tree.",
            len(skills),
        )
        return g

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def nodes(self) -> Dict[str, SkillNode]:
        return self._nodes

    def get(self, name: str) -> Optional[SkillNode]:
        return self._nodes.get(name)

    def roots(self) -> List[SkillNode]:
        """R = all depth-0 nodes."""
        return [n for n in self._nodes.values() if n.is_root()]

    def children_of(self, name: str) -> List[SkillNode]:
        node = self._nodes.get(name)
        if node is None:
            return []
        return [self._nodes[c] for c in node.children if c in self._nodes]

    def ancestors_of(self, name: str) -> List[str]:
        """Return all ancestor names from immediate parent up to root."""
        path: List[str] = []
        node = self._nodes.get(name)
        while node and node.parent:
            path.append(node.parent)
            node = self._nodes.get(node.parent)
        return path

    def all_names(self) -> List[str]:
        return list(self._nodes.keys())

    # ── Observation tracking ──────────────────────────────────────────────────

    def record_observation(self, skill_name: str) -> None:
        """Increment n_s for skill_name."""
        node = self._nodes.get(skill_name)
        if node:
            node.observations += 1

    def observation_count(self, skill_name: str) -> int:
        node = self._nodes.get(skill_name)
        return node.observations if node else 0

    # ── Traversal Policies ────────────────────────────────────────────────────

    def deep_dive(self, current_name: str) -> Optional[str]:
        """
        Deep-Dive Strategy:
            v_{t+1} ∈ children(v_t)

        Selects the child with the fewest observations (most uncertain).
        Returns None if current node has no children.
        """
        children = self.children_of(current_name)
        if not children:
            return None
        # Prefer child with min observations → max uncertainty
        target = min(children, key=lambda n: n.observations)
        logger.debug("Deep-dive: %s → %s", current_name, target.name)
        return target.name

    def topic_switch(self, exclude: Optional[Set[str]] = None) -> Optional[str]:
        """
        Topic-Switch Strategy:
            v_{t+1} = argmin_{r_i ∈ R} n_i

        Selects the root skill that has been assessed the fewest times.
        `exclude` can filter out recently-asked roots to avoid repetition.
        Returns None if no roots exist.
        """
        roots = self.roots()
        if not roots:
            return None
        if exclude:
            candidate_roots = [r for r in roots if r.name not in exclude]
            if not candidate_roots:
                candidate_roots = roots  # fallback: ignore exclusion
        else:
            candidate_roots = roots

        target = min(candidate_roots, key=lambda n: n.observations)
        logger.debug("Topic-switch → %s (n=%d)", target.name, target.observations)
        return target.name

    def next_node(
        self,
        current_name:     str,
        state:            "TraversalState",
        deep_dive_thresh: float = 0.6,
    ) -> Tuple[str, str]:
        """
        State-driven traversal policy:
            v_{t+1} = π(v_t, s_t)

        Decision logic:
            1. If last score was high (≥ deep_dive_thresh) and current node
               has children → Deep-Dive.
            2. Otherwise → Topic-Switch to the least-assessed root.
            3. If neither applies (leaf + low score) → Topic-Switch.

        Returns:
            (next_skill_name, strategy_used)
        """
        node = self._nodes.get(current_name)
        if node is None:
            fallback = self.topic_switch()
            return (fallback or current_name, "topic_switch_fallback")

        # Deep-dive if candidate is performing well and depth is available
        if state.last_score >= deep_dive_thresh and not node.is_leaf():
            child = self.deep_dive(current_name)
            if child:
                return child, "deep_dive"

        # Topic-switch: move to the least-assessed root
        # Exclude the root ancestor of the current node to avoid staying in same domain
        current_root = self.ancestors_of(current_name)
        exclude = {current_root[-1]} if current_root else set()
        switched = self.topic_switch(exclude=exclude)
        if switched:
            return switched, "topic_switch"

        return current_name, "stay"  # last resort

    # ── Utilities ─────────────────────────────────────────────────────────────

    def coverage_rate(self) -> float:
        """Fraction of nodes with at least one observation."""
        if not self._nodes:
            return 0.0
        assessed = sum(1 for n in self._nodes.values() if n.observations > 0)
        return assessed / len(self._nodes)

    def uncovered_roots(self) -> List[str]:
        """Root skills with zero observations."""
        return [r.name for r in self.roots() if r.observations == 0]

    def summary(self) -> Dict:
        return {
            "total_nodes":    len(self._nodes),
            "root_count":     len(self.roots()),
            "coverage_rate":  round(self.coverage_rate(), 3),
            "uncovered_roots": self.uncovered_roots(),
            "nodes": {
                name: {
                    "depth":        n.depth,
                    "weight":       n.weight,
                    "observations": n.observations,
                    "children":     n.children,
                }
                for name, n in self._nodes.items()
            },
        }


# ── TraversalState (carries interview context into π) ────────────────────────

@dataclass
class TraversalState:
    """
    s_t — the state passed into the traversal policy π(v_t, s_t).

    Attributes:
        last_score:          Normalized score of the last answer (0–1).
        recent_skills:       Ordered list of recently asked skill names.
        interview_phase:     Current interview phase string.
        question_count:      Total questions asked so far.
    """
    last_score:      float      = 0.5
    recent_skills:   List[str]  = field(default_factory=list)
    interview_phase: str        = "technical"
    question_count:  int        = 0


# ── Factory helpers ───────────────────────────────────────────────────────────

def build_default_graph(required_skills: List[str]) -> SkillGraph:
    """
    Convenience factory: build a SkillGraph from the job's required_skills.
    If the list is already flat (no '>' separator), all nodes are roots.
    If skills contain '>' (e.g. "分布式系统>Raft"), they are parsed as paths.

    Example:
        skills = ["分布式系统", "分布式系统>Raft", "Redis", "Redis>持久化"]
        → graph with two roots (分布式系统, Redis) and their children.
    """
    g = SkillGraph()
    for skill in required_skills:
        parts = [p.strip() for p in skill.split(">")]
        for i, part in enumerate(parts):
            if part not in g.nodes:
                parent = parts[i - 1] if i > 0 else None
                g.add_node(part, weight=1.0 - i * 0.1, parent=parent)
            if i > 0:
                g.add_edge(parts[i - 1], part)
    return g
