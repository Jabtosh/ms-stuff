from unittest import TestCase

from constants import Q, P21, P, V


class TestConstants(TestCase):
    def test_relationships(self):
        self.assertAlmostEqual(Q[-1], sum(P))
        self.assertAlmostEqual(1, sum(P) + P21)
        self.assertAlmostEqual(1, Q[-1] + P21)
        self.assertEqual(len(V), len(Q))
        self.assertEqual(len(V), len(P))
        for p, q in zip(P.cumsum(), Q):
            self.assertAlmostEqual(p, q)
