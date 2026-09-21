import unittest

from agent.intent_classifier import classify_intent, get_prompt_file


class IntentClassifierTest(unittest.TestCase):
    def test_representative_fitness_queries(self):
        cases = {
            "帮我制定一周三练的增肌计划": "plan",
            "记录今天深蹲60公斤4组": "record",
            "练完膝盖疼应该怎么调整": "adjust",
            "生成九月份训练报告": "report",
            "蛋白质是什么": "qa",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(classify_intent(query), expected)

    def test_unknown_intent_uses_main_prompt(self):
        self.assertEqual(get_prompt_file("unknown"), "main_prompt.txt")


if __name__ == "__main__":
    unittest.main()
