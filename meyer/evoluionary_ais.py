import numpy as np
import numpy.random as ran

from constants import L
from meyer import AiBase
from names import Animals


class EvoAi(AiBase):
    RR_CUTOFF = 5
    NP_CUTOFF = 4

    def __init__(self, doubt_array=None, lie_array=None):
        name = Animals[ran.randint(0, len(Animals))]
        super().__init__(name)
        self.doubt_array = DoubtArray() if doubt_array is None else doubt_array
        self.lie_array = LieArray() if lie_array is None else lie_array
        self.doubt_array.make_compliant()
        self.lie_array.make_compliant()
        self.throw_generator = self.generate_throws()

    def get_name(self) -> str:
        return f"{self.name}: {self.doubt_array.get_identifier():.2f}, {self.lie_array.get_identifier():.2f}"

    @classmethod
    def init_from_data(cls, doubt_array: np.ndarray, lie_array: np.ndarray) -> "EvoAi":
        return cls(DoubtArray.import_(doubt_array), LieArray.import_(lie_array))

    def export_data(self):
        return self.doubt_array.export_(), self.lie_array.export_()

    def doubt_decider(self, claims: tuple, n_players: int, n_rounds_remaining: int) -> bool:
        n_rounds_remaining = n_rounds_remaining if n_rounds_remaining < self.RR_CUTOFF else self.RR_CUTOFF
        n_players_s = n_players - 2 if n_players < self.NP_CUTOFF else self.NP_CUTOFF - 2
        claim_m1_s = claims[-1] - 1
        return self.p_doubt(n_players_s, claims[-2], claim_m1_s, n_rounds_remaining) >= self.throw_generator.__next__()

    def claim_decider(self, claims: tuple, last_throw: int, n_players: int, n_rounds_remaining: int) -> int:
        n_rounds_remaining = n_rounds_remaining if n_rounds_remaining < self.RR_CUTOFF else self.RR_CUTOFF
        n_players = n_players if n_players < self.NP_CUTOFF else self.NP_CUTOFF
        if last_throw > claims[-1]:
            return last_throw
        _p_lie = self.p_lie(n_players - 2, claims[-1] - 1, n_rounds_remaining)
        return np.where(_p_lie.cumsum() > self.throw_generator.__next__())[0][0] + 1

    def p_doubt(self, n_players, claim_m2, claim_m1, n_rounds_remaining) -> float:
        return self.doubt_array[n_rounds_remaining, n_players, claim_m1 - 1, claim_m2]

    def p_lie(self, n_players, claim_m1, n_rounds_remaining) -> np.ndarray:
        return self.doubt_array[n_rounds_remaining, n_players, claim_m1 - 1, :]

    def mutate(self):
        self.doubt_array.mutate()
        self.doubt_array.make_compliant()
        self.lie_array.mutate()
        self.lie_array.make_compliant()

    def reproduce(self, other: "EvoAi"):
        new_doubt_array = self.doubt_array.reproduce(other.doubt_array)
        new_lie_array = self.lie_array.reproduce(other.lie_array)
        return EvoAi(doubt_array=new_doubt_array, lie_array=new_lie_array)

    @staticmethod
    def generate_throws():
        while True:
            random_numbers = ran.random(size=10000)
            for r in random_numbers:
                yield r


class EvoArray(np.ndarray):
    # number of valid outcomes = L - 1
    # M2: 0..19
    # M1: 1..20

    # rounds remaining
    R = EvoAi.RR_CUTOFF + 1
    # number of players - 2
    P = EvoAi.NP_CUTOFF + 1
    # axis 2 = M1 - 1
    # axis 3 = M2
    SHAPE = (R, P, L - 1, L - 1)
    LEN = R * P * (L - 1) * (L - 1)
    STRIDES = (P * (L - 1) * (L - 1), (L - 1) * (L - 1), L - 1, 1)
    INDEX_DATA = np.array(list(filter(lambda x: (x % (L - 1) <= (x // (L - 1)) % (L - 1)), range(LEN))))
    INDEX_REDUNDANT = np.array(list(filter(lambda x: (x % (L - 1) > (x // (L - 1)) % (L - 1)), range(LEN))))

    M_RATE = .05
    M_SCALE = .095

    def __new__(cls, *_, **__):
        """ Initialized with random single precision floating numbers. """
        # TODO: convert existing data to single and adapt __new__
        return np.ndarray.__new__(cls, cls.SHAPE, buffer=ran.random(size=cls.SHAPE))  # .astype(np.single)

    def mutate(self):
        # TODO: act on flat array and cut out compliance check
        mutate = ran.random(size=EvoArray.SHAPE)
        self[mutate < self.M_RATE] += ran.normal(scale=self.M_SCALE, size=self[mutate < self.M_RATE].shape)

    def reproduce(self, other: "EvoArray") -> "EvoArray":
        decider = ran.randint(0, 2)
        new = self.copy()
        new[decider] = other[decider]
        return new

    def get_identifier(self) -> float:
        return np.linalg.norm(self) + np.linalg.norm(self[self < .2]) * 10 - np.linalg.norm(self[self > .75])

    def export_(self) -> np.ndarray:
        return self.flat[self.INDEX_DATA].copy()

    @classmethod
    def import_(cls, data: np.ndarray) -> "EvoArray":
        new = cls(shape=(0,))
        new.flat[cls.INDEX_DATA] = data
        new.make_compliant()
        return new

    def make_compliant(self):
        self.clip(0., 1., out=self)


class DoubtArray(EvoArray):

    def make_compliant(self):
        self.put(self.INDEX_REDUNDANT, 1.)
        super(DoubtArray, self).make_compliant()


class LieArray(EvoArray):

    def make_compliant(self):
        self.put(self.INDEX_REDUNDANT, 0.)
        super(LieArray, self).make_compliant()
        # normalize, such that .sum(axis=3) == 1
        self /= self.sum(axis=3)[:, :, :, np.newaxis]
