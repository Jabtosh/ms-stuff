from unittest import TestCase

from evoluionary_ais import LieArray, DoubtArray


# np.random.permutation(24).reshape(2, 4, 3)

class TestLieArray(TestCase):
    def test_compliance(self):
        ar = LieArray()
        ar.make_compliant()
        self.assertEqual(0, ar[ar > 1].size)
        self.assertEqual(0, ar[ar < 0].size)
        redundant_content = ar.flat[ar.INDEX_REDUNDANT]
        self.assertEqual(0, redundant_content[redundant_content != 0.].size)
        sums = ar.sum(axis=3)
        self.assertEqual(1., sums.mean().item())
        self.assertAlmostEqual(0., sums.std().item(), places=10)


class TestDoubtArray(TestCase):
    def test_compliance(self):
        ar = DoubtArray()
        ar.make_compliant()
        self.assertEqual(0, ar[ar > 1].size)
        self.assertEqual(0, ar[ar < 0].size)
        redundant_content = ar.flat[ar.INDEX_REDUNDANT]
        self.assertEqual(0, redundant_content[redundant_content != 1.].size)
