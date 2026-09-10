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

    def log_message(self, _format, *_args):
        return

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        type(self).calls.append(payload)
        system = payload["messages"][0]["content"]

        if "问题生成器" in system:
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
                question = payload["messages"][1]["content"]
                score = 4 if "简单谈谈" in question else 9
                content = json.dumps({
                    "skill_relevance": score,
                    "difficulty_match": score,
                    "clarity": score,
                    "non_redundancy": score,
                    "contextual_coherence": score,
                    "overall_question_quality": score,
                    "reason": "需要更具体" if score < 7 else "合格",
                }, ensure_ascii=False)
        elif "模拟技术面试候选人" in system:
            content = "我会先说明核心机制，再讨论生产环境中的性能、可靠性和故障恢复权衡。"
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
                    "--question-judge-repeats", "1",
                    "--judge-repeats", "1",
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
                rows = [json.loads(line) for line in turns_path.read_text().splitlines()]
                self.assertEqual(len(rows), 18)
                self.assertTrue(all(row["regeneration_count"] == 1 for row in rows))
                self.assertTrue(all(row["quality_threshold_met"] for row in rows))
                self.assertTrue(all(len(row["question_quality_scores"]) == 5 for row in rows))
                self.assertTrue(all(len(row["question_generation_attempts"]) == 2 for row in rows))
                self.assertIn("question_model_raw_output", rows[0]["question_generation_attempts"][0])
                first_question_judge = rows[0]["question_generation_attempts"][0]["question_judge_details"][0]
                self.assertEqual(first_question_judge["format_retry_count"], 1)
                self.assertEqual(rows[0]["judge_details"][0]["format_retry_count"], 1)
                calls_after_first = len(FakeDashScopeHandler.calls)

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
                        str(ROOT / "analyze_question_quality.py"),
                        "--input", str(turns_path),
                    ],
                    cwd=ROOT, text=True, capture_output=True, check=True,
                )
                analysis_path = turns_path.parent / "quality_analysis" / "question_quality_summary.json"
                report = json.loads(analysis_path.read_text(encoding="utf-8"))
                self.assertEqual(report["n_records"], 18)
                self.assertTrue(report["bridge_vs_random"]["available"])
                self.assertIn("Question quality by strategy", analysis.stdout)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
