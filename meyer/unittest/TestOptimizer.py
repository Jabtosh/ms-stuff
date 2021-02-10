from unittest import TestCase

from meyer_optimizer import Q, p_lie, mu_throw, mu_doubt, mu


class TestOptimizer(TestCase):
    def test_p_lie(self):
        for last_claim in [1, 5, 18, 19]:
            self.assertEqual(Q[last_claim] / (1 - 2 / 36), p_lie(0, last_claim))

    def test_mu_throw(self):
        for (n, last_claim, players_until_me, rounds_remaining, expected) in zip(
                [2, 4, 7, 12, 3, 50], [9, 7, 16, 2, 1, 0], [0, 2, 0, 3, 9, 0], [1, 5, 0, 0, 0, 0],
                [0.0367614026638487, -0.26531612682531436, 0.7592592592592593, -0.020746889631901876,
                 -0.3849898796841011, -0.07210164269456382]):
            self.assertAlmostEqual(expected, mu_throw(n, last_claim, players_until_me, rounds_remaining), places=5)

        for (n, last_claim) in zip([2, 2, 3, 3, 9, 9], [0, 4, 0, 6, 0, 13]):
            # TODO: bugged
            self.assertAlmostEqual(0, sum([mu_throw(n, last_claim, p, 0) for p in range(n)]))
            self.assertAlmostEqual(0, sum([mu_throw(n, last_claim, p, 1) for p in range(n)]), msg="T: rounds remaining")

    def test_mu_doubt(self):
        for (n, claim_m2, players_until_me, rounds_remaining, expected) in zip(
                [2, 4, 7, 12, 3, 50], [1, 3, 1, 4, 10, 0], [0, 2, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0],
                [0.8823529411764706, -0.3997753083569492, 0.9313725490196079, 0.7433155080213905,
                 0.11764705882352944, 1]):
            self.assertAlmostEqual(expected, mu_doubt(n, claim_m2, players_until_me, rounds_remaining), places=5)

        for (n, claim_m2) in zip([2, 2, 3, 3, 9, 9], [0, 4, 0, 6, 0, 13]):
            self.assertAlmostEqual(0, sum([mu_doubt(n, claim_m2, p, 0) for p in range(n)]))
            # TODO: fails because of mu_throw
            self.assertAlmostEqual(0, sum([mu_doubt(n, claim_m2, p, 1) for p in range(n)]), msg="D: rounds remaining")

    def test_mu(self):
        n = 2
        all_ev = [mu(n, 0, 0, p, 0) for p in range(n)]
        self.assertAlmostEqual(0, sum(all_ev))
