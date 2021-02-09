from optimum2021 import Q, p_lie, p_lie_unnecessarily
from unittest import TestCase, patch

class TestOptimizer(TestCase):
    def test_p_lie(self):
        with patch("optimum2021.p_lie_unnecessarily", return_value=0):
            for last_claim in [1, 5, 18, 19]:
                self.assertEqual(Q[last_claim]/(1-2/36), p_lie(3, last_claim))

    #def test_p_lie_unnecessarily(self):
    #    self.assertEqual(0, p_lie_unnecessarily(3, 19))
