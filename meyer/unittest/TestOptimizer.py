from unittest import TestCase

from meyer_optimizer import Q, p_lie, mu_throw, mu_doubt


class TestOptimizer(TestCase):
    def test_p_lie(self):
        for last_claim in [1, 5, 18, 19]:
            self.assertEqual(Q[last_claim] / (1 - 2 / 36), p_lie(0, last_claim))

    def test_mu_throw(self):
        for (n, last_claim, players_until_me, expected) in zip([2, 4, 7, 12, 3, 50], [9, 7, 16, 2, 1, 0],
                                                               [0, 2, 0, 3, 9, 0],
                                                               [2.77555756e-17, -0.16872427983539093,
                                                                0.7592592592592593, -0.020746889631901876,
                                                                -0.3849898796841011, -0.07210164269456382]):
            self.assertAlmostEqual(expected, mu_throw(n, last_claim, players_until_me), places=5)

    def test_mu_doubt(self):
        for (n, claim_m2, expected) in zip([2, 4, 7, 12, 3, 50], [1, 3, 1, 4, 10, 0],
                                           [0.8823529411764706, 0.7647058823529412,
                                            0.9313725490196079, 0.7433155080213905,
                                            0.11764705882352944, 1]):
            self.assertAlmostEqual(expected, mu_doubt(n, claim_m2), places=5)
