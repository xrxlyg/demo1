import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class FakeDashScopeHandler(BaseHTTPRequestHandler):
    calls = []
    generated_questions = 0
    invalid_question_judge_once = True
    invalid_answer_judge_once = True
    invalid_v26_draft_once = True

    def log_message(self, _format, *_args):
        return

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        type(self).calls.append(payload)
        system = payload["messages"][0]["content"]

        if "V2.7证据锚定原子探针规划器" in system or "V2.7技术有效性编辑器" in system:
            prompt = payload["messages"][1]["content"]
            tree = "较低状态：" in prompt or "探针类型：证据锚定" in prompt
            probe = {
                "operation": "predict_outcome",
                "single_task": ("判断旧客户端能否正确解析并说明唯一依据？" if tree
                                else "此次字段变更是否保持向后兼容？"),
                "answer_outline": "忽略新增可选字段，仍能解析原字段。",
                "required_fact_indices": [1],
                "evidence_target": "旧字段语义未变且未知字段可忽略",
                "diagnostic_anchor": "旧客户端忽略未知可选字段，而原有必填字段及其语义均未改变" if tree else "",
                "single_scoring_criterion": "正确判断兼容性并给出支持证据",
                "semantic_checks": {
                    "answer_uses_visible_facts": True,
                    "diagnostic_boundary_preserved": True,
                    "single_scored_product": True,
                    "reason": "唯一结论只依赖可见字段解析规则",
                },
            }
            content = repr(probe) if tree else json.dumps(probe, ensure_ascii=False)
        elif "V2.6共享可见证据契约规划器" in system:
            content = json.dumps({
                "core_concept": "API向后兼容",
                "scenario_text": "服务端在v1响应中新增一个可选字段",
                "stable_facts": [
                    "旧客户端会忽略未知的可选字段",
                    "原有必填字段及其语义均未改变",
                    "响应仍使用原有媒体类型",
                ],
                "answerable_scope": "可判断旧客户端能否继续解析v1响应",
                "assumptions_to_avoid": ["不假设客户端自动升级"],
                "supported_conclusion": "旧客户端仍能解析原有字段",
                "decisive_evidence": "原有必填字段及其语义均未改变",
                "unsupported_claims": ["不能断言任意字段变更都兼容"],
                "diagnostic_boundary": "未知可选字段的解析边界",
                "surface_cue": "响应结构发生变化",
                "decisive_fact_index": 2,
            }, ensure_ascii=False)
        elif "V2.6证据锚定原子探针规划器" in system:
            probe_input = payload["messages"][1]["content"]
            is_tree = "较低状态：" in probe_input
            if is_tree and type(self).invalid_v26_draft_once:
                type(self).invalid_v26_draft_once = False
                task = "依据解析边界判断兼容性并解释原因？"
                anchor = "解析边界"
            elif is_tree:
                task = "依据未知可选字段的解析边界，旧客户端会得到什么结果？"
                anchor = "未知可选字段的解析边界"
            else:
                task = "此次新增可选字段是否保持向后兼容？"
                anchor = ""
            probe_output = {
                "operation": "predict_outcome" if is_tree else "choose_under_constraint",
                "single_task": task,
                "answer_outline": "旧客户端忽略新增字段并继续解析原有字段",
                "required_fact_indices": [1, 2],
                "evidence_target": "原有字段语义不变且未知可选字段可被忽略",
                "answerability_check": "仅依赖可见事实",
                "atomicity_check": "只有一个评分结论",
                "claim_check": "未超出答案边界",
                "diagnostic_anchor": anchor,
                "single_scoring_criterion": "是否正确预测旧客户端解析结果",
                "skill_alignment_check": "直接考查API兼容边界",
                "visible_evidence_check": "答案事实均在题面中可见",
            }
            content = (
                repr(probe_output)
                if is_tree else json.dumps(probe_output, ensure_ascii=False)
            )
        elif "V2.6技术有效性编辑器" in system:
            validation_input = payload["messages"][1]["content"]
            is_tree = "探针类型：证据锚定" in validation_input
            content = json.dumps({
                "operation": "predict_outcome" if is_tree else "choose_under_constraint",
                "single_task": (
                    "依据未知可选字段的解析边界，旧客户端会得到什么结果？"
                    if is_tree else "此次新增可选字段是否保持向后兼容？"
                ),
                "answer_outline": "旧客户端忽略新增字段并继续解析原有字段",
                "required_fact_indices": [1, 2],
                "evidence_target": "原有字段语义不变且未知可选字段可被忽略",
                "answerability_check": "仅依赖可见事实",
                "atomicity_check": "只有一个评分结论",
                "claim_check": "未超出答案边界",
                "diagnostic_anchor": (
                    "未知可选字段的解析边界" if is_tree else ""
                ),
                "single_scoring_criterion": "是否正确判断旧客户端解析结果",
                "skill_alignment_check": "直接考查API兼容边界",
                "visible_evidence_check": "答案事实均在题面中可见",
            }, ensure_ascii=False)
        elif "V2.5共享技术契约规划器" in system:
            content = json.dumps({
                "core_concept": "API向后兼容",
                "scenario_text": (
                    "现有客户端忽略响应中的未知可选字段，本次变更只新增一个可选字段"
                ),
                "stable_facts": [
                    "现有客户端忽略未知可选字段",
                    "本次只新增一个可选响应字段",
                    "原有字段语义保持不变",
                    "旧客户端无需升级",
                    "响应仍使用原有媒体类型",
                ],
                "answerable_scope": "可判断旧客户端对新增字段的解析行为",
                "assumptions_to_avoid": ["不假设客户端自动升级"],
                "supported_conclusion": "旧客户端忽略新增字段并继续解析原有字段",
                "decisive_evidence": "旧客户端对未知可选字段的处理行为",
                "unsupported_claims": ["不能据此断言所有API变更都兼容"],
            }, ensure_ascii=False)
        elif "V2.5原子诊断探针规划器" in system:
            probe_input = payload["messages"][1]["content"]
            is_tree = "较低状态：" in probe_input
            content = json.dumps({
                "operation": (
                    "predict_outcome" if is_tree else "choose_under_constraint"
                ),
                "single_task": (
                    "请推演旧客户端的解析结果、说明原因和回滚方法？"
                    if is_tree else "应选择什么兼容方案并说明理由？"
                ),
                "answer_outline": "旧客户端忽略新增字段并读取原有字段",
                "required_fact_indices": [1, 2, 9],
                "evidence_target": (
                    "客户端解析行为的因果推演" if is_tree else "兼容方案选择"
                ),
                "answerability_check": "仅依赖共享事实",
                "atomicity_check": "草案围绕兼容性",
                "claim_check": "未引入额外事实",
            }, ensure_ascii=False)
        elif "V2.5技术有效性编辑器" in system:
            validation_input = payload["messages"][1]["content"]
            is_tree = "客户端解析行为的因果推演" in validation_input
            content = json.dumps({
                "operation": (
                    "predict_outcome" if is_tree else "identify_decisive_mechanism"
                ),
                "single_task": (
                    "旧客户端收到该响应后会出现什么解析结果？"
                    if is_tree else "哪项机制保证旧客户端仍可解析响应？"
                ),
                "answer_outline": "旧客户端忽略新增字段并读取原有字段",
                "required_fact_indices": [1, 2],
                "evidence_target": (
                    "客户端解析行为的因果推演" if is_tree else "识别兼容机制"
                ),
                "answerability_check": "答案仅使用共享事实",
                "atomicity_check": "只有一个可评分结论",
                "claim_check": "结论位于共享答案边界内",
            }, ensure_ascii=False)
        elif "V2.4共享技术场景规划器" in system:
            content = json.dumps({
                "core_concept": "API向后兼容",
                "scenario_text": (
                    "现有客户端会忽略响应中的未知可选字段，本次变更只新增一个可选"
                    "响应字段；API使用HTTP并保留v1路径，旧客户端无需升级"
                ),
                "stable_facts": [
                    "现有客户端忽略响应中的未知可选字段",
                    "本次变更只新增一个可选响应字段",
                    "API使用HTTP协议",
                    "API继续保留v1路径",
                    "旧客户端无需升级",
                ],
                "answerable_scope": "可判断旧客户端解析行为及此次响应变更的兼容性",
                "assumptions_to_avoid": ["不假设客户端会自动升级"],
            }, ensure_ascii=False)
        elif "V2.4技术面试探针设计器" in system:
            probe_input = payload["messages"][1]["content"]
            is_tree = "较低状态：" in probe_input
            content = json.dumps({
                "single_task": (
                    "请推演旧客户端收到新响应时的解析结果及其兼容性？"
                    if is_tree else "此次响应变更是否向后兼容？"
                ),
                "answer_outline": "旧客户端忽略新增可选字段，因此仍可解析原有字段",
                "required_fact_indices": [1, 2, 9],
                "evidence_target": (
                    "把客户端解析行为与兼容性结论连接为因果链"
                    if is_tree else "识别兼容性"
                ),
                "answerability_check": "结论只依赖两项共享事实",
            }, ensure_ascii=False)
        elif "V2.3技术脚手架规划器" in system:
            content = json.dumps({
                "core_concept": "API向后兼容",
                "stable_facts": [
                    "现有客户端忽略响应中的未知可选字段",
                    "本次变更只新增一个可选响应字段",
                ],
                "probe_options": [
                    {
                        "probe_id": "P1",
                        "single_task": "判断此次变更是否向后兼容并说明原因",
                        "answer_outline": "兼容，因为旧客户端会忽略新增可选字段",
                        "required_fact_indices": [1, 2],
                    },
                    {
                        "probe_id": "P2",
                        "single_task": "指出保证此次变更兼容的决定性条件",
                        "answer_outline": "决定性条件是旧客户端忽略未知可选字段",
                        "required_fact_indices": [1, 2],
                    },
                ],
                "assumptions_to_avoid": ["不假设客户端会自动升级"],
            }, ensure_ascii=False)
        elif "V2.3技术面试问题生成器" in system:
            realization_input = payload["messages"][1]["content"]
            is_tree = "H0：" in realization_input
            content = json.dumps({
                "probe_id": "P2" if is_tree else "P1",
                "question": (
                    "现有客户端会忽略响应中的未知可选字段，本次只新增一个可选响应字段。"
                    + (
                        "保证此次变更向后兼容的决定性条件是什么？"
                        if is_tree else "此次变更是否向后兼容，为什么？"
                    )
                ),
                "answerability_check": "问题明确给出了两个必需事实。",
            }, ensure_ascii=False)
        elif "技术面试问题规划器" in system:
            content = json.dumps({
                "core_concept": "幂等处理",
                "stable_facts": ["消息可能被重复投递"],
                "expected_answer_points": ["使用幂等键避免重复副作用"],
                "single_decision": "判断消费者应如何避免重复副作用",
                "correct_answer_outline": "重复投递需要按业务键去重",
                "assumptions_to_avoid": ["不假设中间件恰好一次投递"],
                "diagnostic_probe": "能否识别业务幂等边界",
                "history_link": "无",
            }, ensure_ascii=False)
        elif "问题生成器" in system:
            type(self).generated_questions += 1
            if type(self).generated_questions % 2:
                content = "请简单谈谈这个技术。"
            else:
                content = f"请结合生产故障解释目标技术的核心机制与权衡（题{type(self).generated_questions}）？"
        elif "问题评审器" in system:
            if type(self).invalid_question_judge_once:
                type(self).invalid_question_judge_once = False
                content = "我认为这个问题不够清晰。"
            else:
                judge_input = payload["messages"][1]["content"]
                question = judge_input.split("候选问题：", 1)[1].split("\n此前对话：", 1)[0]
                score = 4 if ("简单谈谈" in question or "判断旧客户端能否正确解析并说明唯一依据" in question) else 9
                content = json.dumps({
                    "skill_relevance": score,
                    "technical_correctness": score,
                    "difficulty_match": score,
                    "clarity": score,
                    "non_redundancy": score,
                    "contextual_coherence": score,
                    "diagnostic_value": score,
                    "adaptive_relevance": score,
                    "reason": "需要更具体" if score < 7 else "合格",
                }, ensure_ascii=False)
        elif "模拟技术面试候选人" in system:
            content = "我会先说明核心机制，再讨论生产环境中的性能、可靠性和故障恢复权衡。"
        elif "受控的模拟候选人回答" in system:
            content = json.dumps({
                "lower_answer": "我认为收到消息后直接处理即可。",
                "upper_answer": "我会使用业务幂等键记录处理结果，重复消息不再产生副作用。",
            }, ensure_ascii=False)
        elif "盲评诊断区分度评审器" in system:
            content = json.dumps({
                "answer_a_state": "H0",
                "answer_b_state": "H1",
                "question_diagnosticity": 8,
                "confidence": 4,
                "reason": "回答呈现了明确的幂等边界差异",
            }, ensure_ascii=False)
        elif "盲评技术面试问题比较器" in system:
            content = json.dumps({
                "winner": "A",
                "confidence": 4,
                "reason": "A更具诊断性",
            }, ensure_ascii=False)
        elif "技术面试评分器" in system:
            if type(self).invalid_answer_judge_once:
                type(self).invalid_answer_judge_once = False
                content = "候选人的能力评分是6.5分。"
            else:
                content = '{"score": 6.5, "reason": "能力表现中等"}'
        else:
            raise AssertionError(f"Unexpected system prompt: {system}")

        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class QwenPipelineTest(unittest.TestCase):
    def test_three_role_pipeline_and_resume(self):
        FakeDashScopeHandler.calls = []
        FakeDashScopeHandler.generated_questions = 0
        FakeDashScopeHandler.invalid_question_judge_once = True
        FakeDashScopeHandler.invalid_answer_judge_once = True
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeDashScopeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as output:
                command = [
                    sys.executable,
                    str(ROOT / "experiment_all_in_one.py"),
                    "--mode", "qwen",
                    "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--qwen-candidates", "3",
                    "--qwen-questions", "3",
                    "--strategies", "bridge", "random",
                    "--question-judge-repeats", "3",
                    "--quality-threshold", "7",
                    "--max-regenerations", "2",
                    "--request-delay", "0",
                    "--confirm-api-calls",
                    "--output", output,
                ]
                environment = dict(os.environ, DASHSCOPE_API_KEY="test-only-key")
                first = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=True,
                )
                turns_path = Path(output) / "exp3_qwen" / "turns.jsonl"
                rows = [
                    json.loads(line)
                    for line in turns_path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(rows), 18)
                self.assertTrue(all(row["regeneration_count"] == 0 for row in rows))
                self.assertTrue(all(not row["quality_gate_enabled"] for row in rows))
                self.assertEqual(sum(row["quality_threshold_met"] for row in rows), 9)
                self.assertTrue(all(len(row["question_quality_scores"]) == 8 for row in rows))
                self.assertTrue(all(len(row["question_generation_attempts"]) == 1 for row in rows))
                self.assertIn("question_model_raw_output", rows[0]["question_generation_attempts"][0])
                first_question_judge = rows[0]["question_generation_attempts"][0]["question_judge_details"][0]
                self.assertEqual(first_question_judge["format_retry_count"], 1)
                self.assertNotIn("judge_details", rows[0])
                self.assertEqual(rows[0]["ability_observation_source"], "deterministic_latent_simulation")
                self.assertEqual(rows[0]["question_prompt_variant"], "bridge_adaptive")
                random_rows = [row for row in rows if row["strategy"] == "random"]
                self.assertTrue(all(row["question_prompt_variant"] == "baseline" for row in random_rows))
                calls_after_first = len(FakeDashScopeHandler.calls)
                self.assertEqual(calls_after_first, 91)
                systems = [call["messages"][0]["content"] for call in FakeDashScopeHandler.calls]
                self.assertFalse(any("技术面试评分器" in system for system in systems))
                question_judge_inputs = [
                    call["messages"][1]["content"]
                    for call in FakeDashScopeHandler.calls
                    if "问题评审器" in call["messages"][0]["content"]
                ]
                self.assertTrue(all("bridge_adaptive" not in text for text in question_judge_inputs))

                second = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=True,
                )
                self.assertEqual(len(FakeDashScopeHandler.calls), calls_after_first)
                self.assertIn("18/18", second.stderr)
                self.assertIn("Qwen pilot complete", first.stdout)

                analysis = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "experiment_all_in_one.py"),
                        "--mode", "analyze",
                        "--analysis-input", str(turns_path),
                    ],
                    cwd=ROOT, text=True, capture_output=True, check=True,
                )
                analysis_path = turns_path.parent / "quality_analysis" / "question_quality_summary.json"
                report = json.loads(analysis_path.read_text(encoding="utf-8"))
                self.assertEqual(report["n_records"], 18)
                self.assertTrue(report["bridge_vs_random"]["available"])
                self.assertEqual(report["overall"]["quality_gate_enabled_rate"], 0.0)
                self.assertEqual(report["overall"]["below_threshold_rate"], 0.5)
                self.assertIn("Question quality by strategy", analysis.stdout)
        finally:
            server.shutdown()
            server.server_close()

    def test_v22_key_metrics_and_resume(self):
        FakeDashScopeHandler.calls = []
        FakeDashScopeHandler.generated_questions = 0
        FakeDashScopeHandler.invalid_question_judge_once = False
        FakeDashScopeHandler.invalid_answer_judge_once = False
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeDashScopeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as output:
                command = [
                    sys.executable,
                    str(ROOT / "experiment_all_in_one.py"),
                    "--mode", "prompt_ablation_v22",
                    "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--qwen-candidates", "3",
                    "--qwen-questions", "1",
                    "--question-judge-repeats", "1",
                    "--pairwise-judge-repeats", "1",
                    "--diagnostic-discrimination-repeats", "1",
                    "--request-delay", "0",
                    "--confirm-api-calls",
                    "--output", output,
                ]
                environment = dict(os.environ, DASHSCOPE_API_KEY="test-only-key")
                first = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=False,
                )
                self.assertEqual(first.returncode, 0, first.stderr)
                experiment_dir = Path(output) / "exp7_prompt_ablation_v22"
                pairs = [
                    json.loads(line)
                    for line in (experiment_dir / "pairs.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertEqual(len(pairs), 3)
                self.assertEqual(set(pairs[0]["variants"]), {
                    "prompt_plain_v22", "prompt_tree_v22",
                })
                self.assertEqual(len(FakeDashScopeHandler.calls), 36)
                self.assertIn("Blind Tree preference", first.stdout)
                self.assertNotIn('"question_model"', first.stdout)
                key_metrics = json.loads(
                    (experiment_dir / "key_metrics.json").read_text(encoding="utf-8")
                )
                self.assertTrue(key_metrics["available"])
                self.assertIn("technical_correctness", key_metrics)

                calls_after_first = len(FakeDashScopeHandler.calls)
                second = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=True,
                )
                self.assertEqual(len(FakeDashScopeHandler.calls), calls_after_first)
                self.assertIn("3/3", second.stderr)
        finally:
            server.shutdown()
            server.server_close()

    def test_v23_shared_scaffold_and_resume(self):
        FakeDashScopeHandler.calls = []
        FakeDashScopeHandler.generated_questions = 0
        FakeDashScopeHandler.invalid_question_judge_once = False
        FakeDashScopeHandler.invalid_answer_judge_once = False
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeDashScopeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as output:
                command = [
                    sys.executable,
                    str(ROOT / "experiment_all_in_one.py"),
                    "--mode", "prompt_ablation_v23",
                    "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--qwen-candidates", "3",
                    "--qwen-questions", "1",
                    "--question-judge-repeats", "1",
                    "--pairwise-judge-repeats", "1",
                    "--diagnostic-discrimination-repeats", "1",
                    "--request-delay", "0",
                    "--confirm-api-calls",
                    "--output", output,
                ]
                environment = dict(os.environ, DASHSCOPE_API_KEY="test-only-key")
                first = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=False,
                )
                self.assertEqual(first.returncode, 0, first.stderr)
                experiment_dir = Path(output) / "exp8_prompt_ablation_v23"
                pairs = [
                    json.loads(line)
                    for line in (experiment_dir / "pairs.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertEqual(len(pairs), 3)
                self.assertEqual(len(FakeDashScopeHandler.calls), 33)
                for pair in pairs:
                    self.assertEqual(set(pair["variants"]), {
                        "prompt_plain_v23", "prompt_tree_v23",
                    })
                    shared = pair["shared_technical_scaffold"]
                    for generated in pair["variants"].values():
                        attempt = generated["question_generation_attempts"][0]
                        self.assertEqual(attempt["shared_technical_scaffold"], shared)
                        self.assertIn(attempt["selected_probe_id"], {"P1", "P2"})
                    self.assertTrue(
                        pair["pairwise_evaluation"]["strict_technical_gate"]
                    )
                self.assertIn("Blind Tree preference", first.stdout)
                self.assertIn("Estimated planned API calls", first.stdout)

                calls_after_first = len(FakeDashScopeHandler.calls)
                second = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=True,
                )
                self.assertEqual(len(FakeDashScopeHandler.calls), calls_after_first)
                self.assertIn("3/3", second.stderr)
        finally:
            server.shutdown()
            server.server_close()

    def test_v24_shared_facts_distinct_probes_and_resume(self):
        FakeDashScopeHandler.calls = []
        FakeDashScopeHandler.generated_questions = 0
        FakeDashScopeHandler.invalid_question_judge_once = False
        FakeDashScopeHandler.invalid_answer_judge_once = False
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeDashScopeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as output:
                command = [
                    sys.executable,
                    str(ROOT / "experiment_all_in_one.py"),
                    "--mode", "prompt_ablation_v24",
                    "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--qwen-candidates", "3",
                    "--qwen-questions", "1",
                    "--question-judge-repeats", "1",
                    "--pairwise-judge-repeats", "1",
                    "--diagnostic-discrimination-repeats", "1",
                    "--request-delay", "0",
                    "--confirm-api-calls",
                    "--output", output,
                ]
                environment = dict(os.environ, DASHSCOPE_API_KEY="test-only-key")
                first = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=False,
                )
                self.assertEqual(first.returncode, 0, first.stderr)
                experiment_dir = Path(output) / "exp9_prompt_ablation_v24"
                pairs = [
                    json.loads(line)
                    for line in (experiment_dir / "pairs.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertEqual(len(pairs), 3)
                self.assertEqual(len(FakeDashScopeHandler.calls), 33)
                for pair in pairs:
                    self.assertEqual(set(pair["variants"]), {
                        "prompt_plain_v24", "prompt_tree_v24",
                    })
                    scenario = pair["shared_factual_scaffold"]["scenario_text"]
                    self.assertEqual(
                        len(pair["shared_factual_scaffold"]["stable_facts"]), 5
                    )
                    self.assertFalse(
                        pair["shared_factual_scaffold"]["stable_facts_repaired"]
                    )
                    plain_attempt = pair["variants"]["prompt_plain_v24"][
                        "question_generation_attempts"
                    ][0]
                    tree_attempt = pair["variants"]["prompt_tree_v24"][
                        "question_generation_attempts"
                    ][0]
                    self.assertTrue(plain_attempt["question"].startswith(scenario))
                    self.assertTrue(tree_attempt["question"].startswith(scenario))
                    self.assertNotEqual(
                        plain_attempt["single_task"], tree_attempt["single_task"]
                    )
                    self.assertEqual(
                        plain_attempt["shared_factual_scaffold"],
                        tree_attempt["shared_factual_scaffold"],
                    )
                    self.assertEqual(plain_attempt["required_fact_indices"], [1, 2])
                    self.assertEqual(tree_attempt["required_fact_indices"], [1, 2])
                    self.assertTrue(
                        plain_attempt["required_fact_indices_repaired"]
                    )
                    self.assertTrue(
                        tree_attempt["required_fact_indices_repaired"]
                    )
                    self.assertEqual(
                        plain_attempt["invalid_required_fact_indices"], [9]
                    )
                    self.assertTrue(
                        pair["pairwise_evaluation"]["strict_technical_gate"]
                    )
                    for generated in pair["variants"].values():
                        self.assertTrue(
                            generated["counterfactual_discrimination"][
                                "responsive_only"
                            ]
                        )
                self.assertIn("Blind Tree preference", first.stdout)
                self.assertNotIn('"question_model"', first.stdout)
                analysis_report = json.loads(
                    (experiment_dir / "quality_analysis" /
                     "question_quality_summary.json").read_text(encoding="utf-8")
                )
                self.assertIn(
                    "prompt_tree_v24_vs_prompt_plain_v24",
                    analysis_report["paired_comparisons"],
                )

                calls_after_first = len(FakeDashScopeHandler.calls)
                second = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=True,
                )
                self.assertEqual(len(FakeDashScopeHandler.calls), calls_after_first)
                self.assertIn("3/3", second.stderr)
        finally:
            server.shutdown()
            server.server_close()

    def test_v25_contract_validation_and_resume(self):
        FakeDashScopeHandler.calls = []
        FakeDashScopeHandler.generated_questions = 0
        FakeDashScopeHandler.invalid_question_judge_once = False
        FakeDashScopeHandler.invalid_answer_judge_once = False
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeDashScopeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as output:
                command = [
                    sys.executable,
                    str(ROOT / "experiment_all_in_one.py"),
                    "--mode", "prompt_ablation_v25",
                    "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--qwen-candidates", "3",
                    "--qwen-questions", "1",
                    "--question-judge-repeats", "1",
                    "--pairwise-judge-repeats", "1",
                    "--diagnostic-discrimination-repeats", "1",
                    "--request-delay", "0",
                    "--confirm-api-calls",
                    "--output", output,
                ]
                environment = dict(os.environ, DASHSCOPE_API_KEY="test-only-key")
                first = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=False,
                )
                self.assertEqual(first.returncode, 0, first.stderr)
                experiment_dir = Path(output) / "exp10_prompt_ablation_v25"
                pairs = [
                    json.loads(line)
                    for line in (experiment_dir / "pairs.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertEqual(len(pairs), 3)
                self.assertEqual(len(FakeDashScopeHandler.calls), 39)
                for pair in pairs:
                    self.assertEqual(set(pair["variants"]), {
                        "prompt_plain_v25", "prompt_tree_v25",
                    })
                    contract = pair["shared_factual_scaffold"]
                    self.assertEqual(len(contract["stable_facts"]), 5)
                    self.assertIn("supported_conclusion", contract)
                    plain = pair["variants"]["prompt_plain_v25"][
                        "question_generation_attempts"
                    ][0]
                    tree = pair["variants"]["prompt_tree_v25"][
                        "question_generation_attempts"
                    ][0]
                    self.assertEqual(
                        plain["shared_technical_contract"],
                        tree["shared_technical_contract"],
                    )
                    self.assertNotEqual(plain["single_task"], tree["single_task"])
                    self.assertIn("回滚方法", tree["draft_probe"]["single_task"])
                    self.assertNotIn("回滚方法", tree["single_task"])
                    self.assertEqual(tree["single_task"].count("？"), 1)
                    self.assertIn("validation_model_raw_output", tree)
                    self.assertTrue(
                        pair["pairwise_evaluation"]["strict_technical_gate"]
                    )
                self.assertIn("Blind Tree preference", first.stdout)
                self.assertNotIn('"question_model"', first.stdout)
                report = json.loads(
                    (experiment_dir / "quality_analysis" /
                     "question_quality_summary.json").read_text(encoding="utf-8")
                )
                self.assertIn(
                    "prompt_tree_v25_vs_prompt_plain_v25",
                    report["paired_comparisons"],
                )

                calls_after_first = len(FakeDashScopeHandler.calls)
                second = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=True,
                )
                self.assertEqual(len(FakeDashScopeHandler.calls), calls_after_first)
                self.assertIn("3/3", second.stderr)
        finally:
            server.shutdown()
            server.server_close()


    def test_v26_visible_evidence_anchor_and_resume(self):
        FakeDashScopeHandler.calls = []
        FakeDashScopeHandler.generated_questions = 0
        FakeDashScopeHandler.invalid_question_judge_once = False
        FakeDashScopeHandler.invalid_answer_judge_once = False
        FakeDashScopeHandler.invalid_v26_draft_once = True
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeDashScopeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as output:
                command = [
                    sys.executable,
                    str(ROOT / "experiment_all_in_one.py"),
                    "--mode", "prompt_ablation_v26",
                    "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--qwen-candidates", "3",
                    "--qwen-questions", "1",
                    "--question-judge-repeats", "1",
                    "--pairwise-judge-repeats", "1",
                    "--diagnostic-discrimination-repeats", "1",
                    "--request-delay", "0",
                    "--confirm-api-calls",
                    "--output", output,
                ]
                environment = dict(os.environ, DASHSCOPE_API_KEY="test-only-key")
                first = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=False,
                )
                self.assertEqual(first.returncode, 0, first.stderr)
                experiment_dir = Path(output) / "exp11_prompt_ablation_v26"
                pairs = [
                    json.loads(line)
                    for line in (experiment_dir / "pairs.jsonl").read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                self.assertEqual(len(pairs), 3)
                self.assertEqual(len(FakeDashScopeHandler.calls), 40)
                for pair in pairs:
                    self.assertEqual(pair["method_version"], "V2.6")
                    self.assertEqual(set(pair["variants"]), {
                        "prompt_plain_v26", "prompt_tree_v26",
                    })
                    contract = pair["shared_factual_scaffold"]
                    self.assertEqual(contract["decisive_fact_index"], 2)
                    plain = pair["variants"]["prompt_plain_v26"][
                        "question_generation_attempts"
                    ][0]
                    tree = pair["variants"]["prompt_tree_v26"][
                        "question_generation_attempts"
                    ][0]
                    self.assertNotEqual(plain["single_task"], tree["single_task"])
                    self.assertEqual(plain["diagnostic_anchor"], "")
                    self.assertIn(tree["diagnostic_anchor"], tree["single_task"])
                    self.assertNotIn("并解释", tree["single_task"])
                    self.assertTrue(plain["all_contract_facts_visible"])
                    self.assertTrue(tree["all_contract_facts_visible"])
                    for fact in contract["stable_facts"]:
                        self.assertIn(fact, plain["question"])
                        self.assertIn(fact, tree["question"])
                    self.assertTrue(
                        pair["pairwise_evaluation"]["strict_technical_gate"]
                    )
                self.assertIn("Blind Tree preference", first.stdout)
                self.assertNotIn('"question_model"', first.stdout)
                report = json.loads(
                    (experiment_dir / "quality_analysis" /
                     "question_quality_summary.json").read_text(encoding="utf-8")
                )
                self.assertIn(
                    "prompt_tree_v26_vs_prompt_plain_v26",
                    report["paired_comparisons"],
                )

                calls_after_first = len(FakeDashScopeHandler.calls)
                second = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True,
                    capture_output=True, check=True,
                )
                self.assertEqual(len(FakeDashScopeHandler.calls), calls_after_first)
                self.assertIn("3/3", second.stderr)
        finally:
            server.shutdown()
            server.server_close()


class V27ValidationTest(unittest.TestCase):
    def setUp(self):
        import experiment_all_in_one as experiment
        self.m = experiment
        self.scaffold = {
            "core_concept": "隐私控制",
            "stable_facts": ["备份权限为所有用户可读", "备份未加密"],
            "decisive_fact_index": 2,
            "supported_conclusion": "权限设置和未加密使备份数据暴露",
            "diagnostic_boundary": "从可见访问与加密证据识别隐私控制缺陷",
        }
        self.probe = {
            "operation": "identify_decisive_mechanism",
            "single_task": "识别数据暴露的控制缺陷并给出支持证据？",
            "answer_outline": "所有用户可读且备份未加密，导致数据暴露。",
            "required_fact_indices": [1],
            "evidence_target": "权限与加密事实",
            "diagnostic_anchor": "备份文件未加密且权限设置为所有用户可读；该证据描述允许自然概括，不能要求逐字抄写到问题中。" * 3,
            "single_scoring_criterion": "正确识别并论证核心隐私控制缺陷",
        }

    def parse(self, probe=None, **kwargs):
        return self.m.parse_v27_probe_json(
            json.dumps(self.probe if probe is None else probe, ensure_ascii=False),
            2, True, self.scaffold, **kwargs,
        )

    def test_standard_json_and_long_nonverbatim_anchor(self):
        result = self.parse()
        self.assertEqual(result["diagnostic_anchor"], self.probe["diagnostic_anchor"])
        self.assertFalse(result["diagnostic_anchor_verbatim_in_task"])
        self.assertEqual(result["single_task"], self.probe["single_task"])

    def test_safe_python_literal_and_no_code_execution(self):
        result = self.m.parse_v27_probe_json(repr(self.probe), 2, True, self.scaffold)
        self.assertEqual(result["answer_outline"], self.probe["answer_outline"])
        with self.assertRaises(self.m.JsonFormatError):
            self.m.decode_probe_object("{'x': __import__('os').getcwd()}")

    def test_missing_audit_index_repaired_without_call_or_question_change(self):
        result = self.parse()
        self.assertEqual(result["decisive_fact_index"], 2)
        self.assertEqual(result["required_fact_indices"], [1, 2])
        self.assertTrue(result["repaired"])
        self.assertIn("decisive_fact_index_from_shared_contract", result["repair_log"])
        self.assertEqual(result["single_task"], self.probe["single_task"])

    def test_evidence_supported_single_products_accepted(self):
        for task in ("判断结论并说明唯一依据？", "选择一个方案并说明核心权衡？",
                     "识别机制并给出支持证据？", "识别并论证核心隐私控制缺陷？",
                     "判断并说明依据？", "依据现有修复方案判断故障原因？"):
            with self.subTest(task=task):
                self.parse({**self.probe, "single_task": task})

    def test_independent_products_rejected(self):
        for task in ("设计算法并分析复杂度？", "诊断原因并给出修复方案？", "根因诊断 + 修复方案？",
                     "选择版本并制定回滚计划？", "列出原因并给出排查步骤及验证点？",
                     "分析故障传播并设计独立恢复策略？",
                     "Give a root cause diagnosis and propose a remediation plan?"):
            with self.subTest(task=task), self.assertRaisesRegex(self.m.ProbeValidationError, "两个可独立评分"):
                self.parse({**self.probe, "single_task": task})

    def test_claim_strength_and_semantic_editor_failures(self):
        self.scaffold["supported_conclusion"] = "可能是权限设置问题，需验证"
        with self.assertRaisesRegex(self.m.ProbeValidationError, "claim_strength"):
            self.parse({**self.probe, "single_task": "根本原因是什么？"})
        checks = {"answer_uses_visible_facts": True, "diagnostic_boundary_preserved": True,
                  "single_scored_product": True, "reason": "reference answer invents an invisible log"}
        for key in ("answer_uses_visible_facts", "diagnostic_boundary_preserved", "single_scored_product"):
            with self.subTest(rule=key), self.assertRaisesRegex(self.m.ProbeValidationError, key):
                self.parse({**self.probe, "semantic_checks": {**checks, key: False}}, require_semantic_checks=True)

    def client(self, outputs):
        class FakeClient:
            def __init__(self):
                self.outputs = iter(outputs)
                self.users = []
            def chat(self, model, system, user, **kwargs):
                self.users.append(user)
                return next(self.outputs)
        return FakeClient()

    def test_truncated_json_retries_with_specific_error(self):
        raw = json.dumps(self.probe, ensure_ascii=False)
        client = self.client([raw[:-25], raw])
        result = self.m.request_judge_json(client, "fake", "system", "task",
                     lambda raw: self.m.parse_v27_probe_json(raw, 2, True, self.scaffold))
        self.assertEqual(result["format_retry_count"], 1)
        self.assertEqual(result["invalid_raw_outputs"][0]["error_category"], "json_format")
        self.assertIn("JsonFormatError", client.users[1])
        self.assertIn("format_attempt=1", client.users[1])
        self.assertIn("missing or truncated JSON object", client.users[1])

    def test_semantic_retry_reason_and_raw_journal(self):
        invalid = json.dumps({**self.probe, "single_task": "诊断原因并给出修复方案？"}, ensure_ascii=False)
        client = self.client([invalid, json.dumps(self.probe, ensure_ascii=False)])
        with tempfile.TemporaryDirectory() as directory:
            client.json_audit_path = Path(directory) / "json_requests.jsonl"
            result = self.m.request_judge_json(client, "fake", "system", "task",
                         lambda raw: self.m.parse_v27_probe_json(raw, 2, True, self.scaffold))
            records = [json.loads(line) for line in client.json_audit_path.read_text().splitlines()]
        self.assertEqual(result["invalid_raw_outputs"][0]["error_category"], "semantic_validation")
        self.assertIn("根因诊断 + 修复方案", client.users[1])
        self.assertEqual(records[0]["raw"], invalid)
        self.assertEqual(records[1]["status"], "accepted")

    def test_terminal_error_retains_rule_attempt_and_full_output(self):
        invalid = json.dumps({**self.probe, "single_task": "诊断原因并给出修复方案？"}, ensure_ascii=False)
        client = self.client([invalid] * 3)
        with self.assertRaises(self.m.JsonRequestError) as caught:
            self.m.request_judge_json(client, "fake", "system", "task",
                     lambda raw: self.m.parse_v27_probe_json(raw, 2, True, self.scaffold))
        message = str(caught.exception)
        for value in ("format_attempt=3/3", "semantic_validation", "ProbeValidationError", "修复方案", self.probe["diagnostic_anchor"]):
            self.assertIn(value, message)
        self.assertEqual(len(caught.exception.invalid_raw_outputs), 3)

    def test_v26_parser_retains_original_strict_rules(self):
        with self.assertRaises(ValueError):
            self.m.parse_v26_probe_json(json.dumps(self.probe, ensure_ascii=False), 2, True, self.scaffold)

    def test_v27_fake_api_3x3_budget_analysis_retention_and_resume(self):
        FakeDashScopeHandler.calls = []
        FakeDashScopeHandler.generated_questions = 0
        FakeDashScopeHandler.invalid_question_judge_once = False
        FakeDashScopeHandler.invalid_answer_judge_once = False
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeDashScopeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as output:
                command = [sys.executable, str(ROOT / "experiment_all_in_one.py"),
                    "--mode", "prompt_ablation_v27", "--seed", "20260919",
                    "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                    "--qwen-candidates", "3", "--qwen-questions", "3",
                    "--question-judge-repeats", "3", "--pairwise-judge-repeats", "3",
                    "--diagnostic-discrimination-repeats", "1",
                    "--request-delay", "0", "--confirm-api-calls", "--output", output]
                env = dict(os.environ, DASHSCOPE_API_KEY="test-only-key")
                first = subprocess.run(command, env=env, text=True, capture_output=True)
                self.assertEqual(first.returncode, 0, first.stderr)
                self.assertEqual(len(FakeDashScopeHandler.calls), 171)
                draft_calls = [call for call in FakeDashScopeHandler.calls
                               if "V2.7证据锚定原子探针规划器" in call["messages"][0]["content"]]
                self.assertEqual(len(draft_calls), 18)
                for index in range(0, 18, 2):
                    histories = [call["messages"][1]["content"].split("共享同叶子历史：", 1)[1].split("\n", 1)[0]
                                 for call in draft_calls[index:index + 2]]
                    self.assertEqual(histories[0], histories[1])
                    self.assertTrue(all(call["model"] == "qwen-turbo" for call in draft_calls[index:index + 2]))
                for call in FakeDashScopeHandler.calls:
                    if "V2.7技术有效性编辑器" in call["messages"][0]["content"]:
                        self.assertEqual(call["model"], "qwen-turbo")
                directory = Path(output) / "exp12_prompt_ablation_v27"
                original_pairs = (directory / "pairs.jsonl").read_bytes()
                pairs = [json.loads(line) for line in original_pairs.splitlines()]
                self.assertEqual(len(pairs), 9)
                for pair in pairs:
                    for variant, generated in pair["variants"].items():
                        self.assertEqual(generated["regeneration_count"], 0)
                        attempt = generated["question_generation_attempts"][0]
                        self.assertTrue(attempt["all_contract_facts_visible"])
                        self.assertEqual(len(generated["question_generation_attempts"]), 1)
                        if "tree" in variant:
                            audit = attempt["final_probe_audit"]
                            self.assertFalse(audit["diagnostic_anchor_verbatim_in_task"])
                            self.assertTrue(audit["diagnostic_fact_index_repaired"])
                            self.assertIn(2, audit["required_fact_indices"])
                        for fact in pair["shared_factual_scaffold"]["stable_facts"]:
                            self.assertIn(fact, generated["question"])
                summary = json.loads((directory / "summary.json").read_text())
                # Deliberately poor Tree questions remain in all records and summaries.
                self.assertEqual(summary["validity_audit"]["prompt_tree_v27"]["technical_correctness_below_7_rate"], 1)
                self.assertLess(summary["paired_absolute_quality"]["metrics"]["technical_correctness"]["mean_delta"], 0)
                self.assertEqual(summary["pair_diagnostics"]["all_contract_facts_visible_rate"], 1)
                self.assertEqual(summary["pair_diagnostics"]["probe_task_exact_match_rate"], 0)
                for text in ("Blind Tree preference", "facts-visible=100.00%", "task-collapse=0.00%", "171"):
                    self.assertIn(text, first.stdout)
                self.assertNotIn('"question_model"', first.stdout)
                # A complete identical rerun makes no new calls or pair writes.
                second = subprocess.run(command, env=env, text=True, capture_output=True)
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertEqual(len(FakeDashScopeHandler.calls), 171)
                self.assertEqual((directory / "pairs.jsonl").read_bytes(), original_pairs)
                self.assertIn("9/9", second.stderr)
                # Analyze is also independent of API access.
                analysis = subprocess.run([sys.executable, str(ROOT / "experiment_all_in_one.py"),
                    "--mode", "analyze", "--analysis-input", str(directory / "turns.jsonl"),
                    "--analysis-output", str(directory / "analysis_cli")],
                    text=True, capture_output=True)
                self.assertEqual(analysis.returncode, 0, analysis.stderr)
                report = json.loads((directory / "analysis_cli" / "question_quality_summary.json").read_text())
                self.assertIn("prompt_tree_v27_vs_prompt_plain_v27", report["paired_comparisons"])
                verbose = subprocess.run(command + ["--verbose-results"], env=env, text=True, capture_output=True)
                self.assertEqual(verbose.returncode, 0, verbose.stderr)
                self.assertIn('"question_model"', verbose.stdout)
                self.assertEqual(len(FakeDashScopeHandler.calls), 171)
                # Simulate interruption after the first candidate's three complete pairs.
                prefix = b"".join(original_pairs.splitlines(keepends=True)[:3])
                (directory / "pairs.jsonl").write_bytes(prefix)
                partial = subprocess.run(command, env=env, text=True, capture_output=True)
                self.assertEqual(partial.returncode, 0, partial.stderr)
                self.assertEqual(len(FakeDashScopeHandler.calls), 171 + 6 * 19)
                self.assertTrue((directory / "pairs.jsonl").read_bytes().startswith(prefix))
                self.assertEqual(len((directory / "pairs.jsonl").read_text().splitlines()), 9)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
