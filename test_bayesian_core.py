"""Regression tests for the Bayesian ability model and risk selector."""

import unittest

from ability_model import AbilityModel
from question_selector import QuestionSelector, score_all


class _Node:
    def __init__(self, name, parent=None, depth=0, weight=1.0):
        self.name = name
        self.parent = parent
        self.depth = depth
        self.weight = weight
        self.children = []
        self.observations = 0


class _Graph:
    def __init__(self):
        self.nodes = {
            name: _Node(name)
            for name in ("backend", "redis", "mysql", "frontend")
        }
        for child in ("redis", "mysql"):
            self.nodes[child].parent = "backend"
            self.nodes[child].depth = 1
        self.nodes["backend"].children = ["redis", "mysql"]

    def roots(self):
        return [node for node in self.nodes.values() if node.parent is None]

    def children_of(self, name):
        return [self.nodes[child] for child in self.nodes[name].children]

    def all_names(self):
        return list(self.nodes)

    def get(self, name):
        return self.nodes.get(name)

    def topic_switch(self, exclude=None):
        return "backend"

    def next_node(self, current, state):
        return "redis", "deep_dive"


class BayesianCoreTest(unittest.TestCase):
    def setUp(self):
        self.graph = _Graph()
        self.model = AbilityModel(prior_std=0.30, observation_std=0.10)
        self.model.initialize_from_job_analysis({
            "required_skills": self.graph.all_names(),
            "competency_dimensions": ["communication"],
        })
        self.assertTrue(self.model.configure_skill_graph(self.graph))

    def test_exact_scalar_posterior(self):
        self.model.update_skill("frontend", 8.0)
        # prior N(.5, .09), likelihood N(.8, .01)
        self.assertAlmostEqual(self.model.skills["frontend"].mean, 0.77, places=6)
        self.assertAlmostEqual(self.model.skills["frontend"].variance, 0.009, places=6)

    def test_graph_propagates_evidence(self):
        before = self.model.skills["mysql"].mean
        self.model.update_skill("redis", 8.0)
        self.assertGreater(self.model.skills["mysql"].mean, before)
        self.assertEqual(self.model.skills["mysql"].observations, 0)

    def test_disagreement_controls_update_strength(self):
        low_noise = AbilityModel(prior_std=0.30, observation_std=0.10)
        high_noise = AbilityModel(prior_std=0.30, observation_std=0.10)
        low_noise.update_skill("redis", 8.0, score_std=0.2)
        high_noise.update_skill("redis", 8.0, score_std=2.0)
        self.assertGreater(low_noise.skills["redis"].mean, high_noise.skills["redis"].mean)
        self.assertLess(low_noise.skills["redis"].variance, high_noise.skills["redis"].variance)

    def test_risk_selector_returns_auditable_components(self):
        self.model.update_skill("redis", 8.0)
        rows = score_all(
            self.model,
            self.graph.all_names(),
            "technical",
            "medium",
            [],
            skill_graph=self.graph,
        )
        self.assertEqual(len(rows), 4)
        self.assertIn("decision_reduction", rows[0])
        self.assertIn("global_reduction", rows[0])
        target = QuestionSelector().select(
            self.model,
            self.graph.all_names(),
            "technical",
            "medium",
            [],
        )
        self.assertIn(target.target, self.graph.all_names())


if __name__ == "__main__":
    unittest.main()
