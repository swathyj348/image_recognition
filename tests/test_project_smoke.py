import unittest
from pathlib import Path

from wine_agent.agent import WineAnalystAgent
from image_agent.reverse_search import search_similar_images


class ProjectSmokeTests(unittest.TestCase):
    def test_wine_agent_reads_outputs(self):
        agent = WineAnalystAgent('wine_agent/outputs')
        answer = agent.answer('What chemical factors most strongly predict high-quality wine?')
        self.assertIn('validation', answer.lower())
        self.assertIn('predict', answer.lower())

    def test_reverse_search_smoke_index_exists(self):
        index_path = Path('image_agent/reverse_index_smoke.npz')
        self.assertTrue(index_path.exists())
        results = search_similar_images('image_dataset/Apple/img_01.jpeg', index_path, top_k=5)
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]['class'], 'Apple')


if __name__ == '__main__':
    unittest.main()
