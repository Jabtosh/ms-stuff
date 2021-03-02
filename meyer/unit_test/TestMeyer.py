from constants import V, P21
from meyer import Game, AiBase
from unit_test.ExtendedTestCase import ExtendedTestCase


class DoubtAi(AiBase):

    def doubt_decider(self, claims: tuple, n_players: int, n_rounds_remaining: int) -> bool:
        return True

    def claim_decider(self, claims: tuple, last_throw: int, n_players: int, n_rounds_remaining: int) -> int:
        return V[-1]


class BelieveAi(DoubtAi):

    def doubt_decider(self, claims: tuple, n_players: int, n_rounds_remaining: int) -> bool:
        return False


class TestMeyer(ExtendedTestCase):
    def test_move_invalid(self):
        game = Game([DoubtAi("1"), DoubtAi("2")], 10)
        self.assertFalse(game.move_invalid())
        for claim_m2 in V:
            for claim_m1 in V:
                game.claims = (0, claim_m2, claim_m1)
                self.assertEqual(claim_m1 <= claim_m2, game.move_invalid())
        for claim_m1 in range(30):
            game.claims = (0, 20, claim_m1)
            self.assertTrue(game.move_invalid())

    def test_throw_generator(self):
        game = Game([DoubtAi("1"), DoubtAi("2")], 10)
        returned_values = set()
        expected = set(V[1:])
        expected.add(21)
        while returned_values < expected:
            returned_values.add(game.throw_generator.__next__())
        self.assertSetEqual(expected, returned_values)
