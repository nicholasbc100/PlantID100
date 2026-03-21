import unittest

from plant_id import PlantIDConfig, PlantIDEngine, PlantReference


def _references():
    return [
        PlantReference("acer_rubrum", (0.2, 0.8, 0.1), 1.8, 0.7, (0.18, 0.62, 0.18)),
        PlantReference("ficus_benjamina", (0.7, 0.3, 0.5), 2.4, 0.1, (0.20, 0.52, 0.19)),
        PlantReference("quercus_robur", (0.3, 0.7, 0.2), 1.5, 0.5, (0.22, 0.56, 0.16)),
    ]

class PlantIDEngineTests(unittest.TestCase):
    def test_identify_returns_sorted_candidates(self):
        engine = PlantIDEngine(_references(), PlantIDConfig(top_k=3))
        results = engine.identify(
            image_embedding=(0.25, 0.75, 0.15),
            leaf_aspect_ratio=1.65,
            serration_score=0.62,
            color_profile=(0.20, 0.58, 0.17),
        )

        self.assertGreaterEqual(len(results), 1)
        confidences = [candidate.confidence for candidate in results]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertIn(results[0].species_id, {"acer_rubrum", "quercus_robur"})

    def test_identify_respects_top_k_and_confidence_filter(self):
        engine = PlantIDEngine(
            _references(),
            PlantIDConfig(top_k=1, min_confidence=0.9),
        )
        results = engine.identify(
            image_embedding=(0.25, 0.75, 0.15),
            leaf_aspect_ratio=1.65,
            serration_score=0.62,
            color_profile=(0.20, 0.58, 0.17),
        )

        self.assertLessEqual(len(results), 1)
        if results:
            self.assertGreaterEqual(results[0].confidence, 0.9)
