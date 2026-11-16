import unittest

from probability_utils import probability_at_least


class ProbabilityUtilsTests(unittest.TestCase):
    def test_probability_is_bounded(self):
        for mean in (0, 1, 2, 3, 5, 10):
            value = probability_at_least(mean, 1.23, 3)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_probability_increases_with_projection(self):
        values = [probability_at_least(mean, 1.23, 3) for mean in (1, 2, 3, 4, 5)]
        self.assertEqual(values, sorted(values))

    def test_probability_decreases_for_higher_threshold(self):
        values = [probability_at_least(3.0, 1.23, cutoff) for cutoff in (2, 3, 4)]
        self.assertEqual(values, sorted(values, reverse=True))


if __name__ == "__main__":
    unittest.main()
