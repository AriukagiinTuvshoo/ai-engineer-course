"""Offline unit tests for autonomous_agent_v4.py. No Gemini API calls are made."""
import unittest
import autonomous_agent_v4 as agent


class TestAgentV4(unittest.TestCase):
    def test_detect_language_mongolian(self):
        self.assertEqual(agent.detect_language("Файлыг уншаад тайлбарла"), "mn")

    def test_detect_language_english(self):
        self.assertEqual(agent.detect_language("Read and summarize this file"), "en")

    def test_safe_calculate(self):
        self.assertEqual(agent.safe_calculate("1250 * 24 + 3600"), 33600)

    def test_reject_unsafe_expression(self):
        with self.assertRaises(ValueError):
            agent.safe_calculate("__import__('os').system('whoami')")

    def test_reject_non_arithmetic(self):
        self.assertIsNone(agent.extract_arithmetic("Make a 7 day study plan with 60 minutes per day"))

    def test_detect_explicit_arithmetic(self):
        self.assertEqual(agent.extract_arithmetic("Calculate 1250 * 24 + 3600"), "1250*24+3600")

    def test_real_file_read(self):
        agent.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        path = agent.KNOWLEDGE_DIR / "v4_test_sample.txt"
        try:
            path.write_text("AI agents can use tools.", encoding="utf-8")
            self.assertEqual(agent.read_knowledge_file("v4_test_sample.txt"), "AI agents can use tools.")
        finally:
            if path.exists():
                path.unlink()

    def test_reject_path_traversal(self):
        with self.assertRaises(ValueError):
            agent.read_knowledge_file("../ai_memory.db")


if __name__ == "__main__":
    unittest.main(verbosity=2)
