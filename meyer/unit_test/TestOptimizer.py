from constants import V
from meyer_optimizer import Q, p_lie, mu_throw, mu_doubt, mu
from unit_test.ExtendedTestCase import ExtendedTestCase

N_MAX = 10


class TestOptimizer(ExtendedTestCase):
    def test_p_lie(self):
        for last_claim in [1, 5, 18, 19]:
            self.assertEqual(Q[last_claim] / (1 - 2 / 36), p_lie(0, last_claim))

    def test_mu(self):
        for n in range(2, N_MAX):
            all_ev = [mu(n, 0, 0, p, 0) for p in range(n)]
            self.assertAlmostEqual(0, sum(all_ev))

    def test_mu_doubt(self):
        for n in range(2, N_MAX):
            for v in V:
                self.assertAlmostEqual(0, sum(mu_doubt(n, v, p, 0) for p in range(n)), msg=f"n={n}, v={v}")

    def test_mu_throw(self):
        for n in range(2, N_MAX):
            for v in V:
                self.assertAlmostEqual(0, sum(mu_throw(n, v, p, 0) for p in range(n)), msg=f"n={n}, v={v}")

        n = 2
        self.assertSeqAlmostEqual([0.8888888888888888, -0.8888888888888888],
                                  [mu_throw(n, V[20], p, 0) for p in range(n)])
        n = 4
        self.assertSeqAlmostEqual([0.8888888888888888, -0.2962962962962963, -0.2962962962962963, -0.2962962962962963],
                                  [mu_throw(n, V[20], p, 0) for p in range(n)])
        n = 3
        self.assertSeqAlmostEqual([0.8888888888888888, -0.4444444444444444, -0.4444444444444444],
                                  [mu_throw(n, V[20], p, 0) for p in range(n)])
        n = 2
        self.assertSeqAlmostEqual([0.8888888888888888, -0.8888888888888888],
                                  [mu_throw(n, V[20], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.8333333333333333, -0.8333333333333333],
                                  [mu_throw(n, V[19], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.7777777777777777, -0.7777777777777777],
                                  [mu_throw(n, V[18], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.7222222222222222, -0.7222222222222222],
                                  [mu_throw(n, V[17], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.6666666666666667, -0.6666666666666667],
                                  [mu_throw(n, V[16], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.6111111111111112, -0.6111111111111112],
                                  [mu_throw(n, V[15], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.5555555555555556, -0.5555555555555556],
                                  [mu_throw(n, V[14], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.4444444444444444, -0.4444444444444444],
                                  [mu_throw(n, V[13], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.33333333333333326, -0.33333333333333326],
                                  [mu_throw(n, V[12], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.22222222222222227, -0.22222222222222227],
                                  [mu_throw(n, V[11], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.11111111111111116, -0.11111111111111116],
                                  [mu_throw(n, V[10], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([0.0, 0.0],
                                  [mu_throw(n, V[9], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([-0.10784313725490197, 0.10784313725490197],
                                  [mu_throw(n, V[8], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([-0.1929920116194625, 0.1929920116194625],
                                  [mu_throw(n, V[7], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([-0.25343944162026955, 0.25343944162026955],
                                  [mu_throw(n, V[6], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([-0.2905577297219756, 0.2905577297219756],
                                  [mu_throw(n, V[5], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([-0.3056429393634199, 0.3056429393634199],
                                  [mu_throw(n, V[4], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([-0.2989205828918865, 0.2989205828918865],
                                  [mu_throw(n, V[3], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([-0.27187452291278025, 0.27187452291278025],
                                  [mu_throw(n, V[2], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([-0.22636007047790177, 0.22636007047790177],
                                  [mu_throw(n, V[1], p, 0) for p in range(n)])
        self.assertSeqAlmostEqual([-0.1649464572668927, 0.1649464572668927],
                                  [mu_throw(n, V[0], p, 0) for p in range(n)])
