from unittest import TestCase

from meyer_optimizer import Q, p_lie, mu_throw, mu_doubt, mu


class TestOptimizer(TestCase):
    def test_p_lie(self):
        for last_claim in [1, 5, 18, 19]:
            self.assertEqual(Q[last_claim] / (1 - 2 / 36), p_lie(0, last_claim))

    def test_mu_throw(self):
        for (n, last_claim, players_until_me, rounds_remaining, expected) in zip(
                [2, 4, 7, 12, 3, 50], [9, 7, 16, 2, 1, 0], [0, 2, 0, 3, 9, 0], [1, 5, 0, 0, 0, 0],
                [0.0367614026638487, -0.40445498336022223, 0.7592592592592593, -0.0656298758729128,
                 -0.3849898796841011, 0.12709840769596273]):
            self.assertAlmostEqual(expected, mu_throw(n, last_claim, players_until_me, rounds_remaining), places=5)

    def test_mu_doubt(self):
        for (n, claim_m2, players_until_me, rounds_remaining, expected) in zip(
                [2, 4, 7, 12, 3, 50], [1, 3, 1, 4, 10, 0], [0, 2, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0],
                [0.8823529411764706, -0.514875055745836, 0.9313725490196079, 0.7433155080213905,
                 0.11764705882352944, 1]):
            self.assertAlmostEqual(expected, mu_doubt(n, claim_m2, players_until_me, rounds_remaining), places=5)

    def test_mu(self):
        n = 4
        all_ev = [mu(n, 0, 0, p, 5) for p in range(n)]
        self.assertAlmostEqual(0, sum(all_ev))
