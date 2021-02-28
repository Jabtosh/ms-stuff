import numpy as np
import numpy.random as ran

from constants import L, V
from meyer import AiBase


class EvoAi(AiBase):
    RR_CUTOFF = 5
    NP_CUTOFF = 4

    def __init__(self, name):
        super().__init__(name)
        self.throw_generator = self.generate_throws()

    def doubt_decider(self, claims: tuple, n_players: int, n_rounds_remaining: int) -> bool:
        n_rounds_remaining = n_rounds_remaining if n_rounds_remaining < self.RR_CUTOFF else self.RR_CUTOFF
        n_players = n_players if n_players < self.NP_CUTOFF else self.NP_CUTOFF
        claim_m2 = claims[-2] if len(claims) > 1 else 0
        claim_m1 = claims[-1]
        # TODO: move this rule set into game; init claims with 0
        if claim_m2 == claim_m1 == 0:
            return False
        elif claim_m1 not in V[claim_m2 + 1:]:
            return True
        return self.p_doubt(n_players, claim_m2, claim_m1, n_rounds_remaining) >= self.throw_generator.__next__()

    def claim_decider(self, claims: tuple, last_throw: int, n_players: int, n_rounds_remaining: int) -> int:
        n_rounds_remaining = n_rounds_remaining if n_rounds_remaining < self.RR_CUTOFF else self.RR_CUTOFF
        n_players = n_players if n_players < self.NP_CUTOFF else self.NP_CUTOFF
        if not claims or last_throw > claims[-1]:
            return last_throw
        _p_lie = self.p_lie(n_players, claims[-1], n_rounds_remaining)
        return np.where(_p_lie.cumsum() > self.throw_generator.__next__())[0][0]

    def p_doubt(self, n_players, claim_m2, claim_m1, n_rounds_remaining) -> float:
        pass

    def p_lie(self, n_players, claim_m1, n_rounds_remaining) -> np.ndarray:
        pass

    @staticmethod
    def generate_throws():
        while True:
            random_numbers = ran.random(size=10000)
            for r in random_numbers:
                yield r


# RR, NP, M1, M2
class EvoArray(np.ndarray):
    R = EvoAi.RR_CUTOFF
    P = EvoAi.NP_CUTOFF
    L = L - 1
    SHAPE = (R, P, L, L - 1)

    M_RATE = .05
    M_SCALE = .08

    def __new__(cls, *args, **kwargs):
        return ran.random(size=cls.SHAPE)

    def mutate(self):
        mutate = ran.random(size=self.SHAPE)
        mutation = ran.normal(scale=self.M_SCALE, size=self.SHAPE)
        self[mutate > self.M_RATE] += mutation

    def reproduce(self, other):
        decider = ran.randint(0, 2)
        self[decider] = other[decider]


class DoubtArray(EvoArray):

    def make_compliant(self):
        for i in range(self.L):
            self[:, :, i, :i] = 1.
        self[self > 1.] = 1.


class LieArray(EvoArray):
    # np.random.permutation(24).reshape(2, 4, 3)

    def make_compliant(self):
        for i in range(self.L):
            self[:, :, i, :i] = 1.
        # normalize, such that .sum(axis=3) == 1
        self /= self.sum(axis=3)[:, :, :, np.newaxis]
